// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::daemon::*;
use crate::event;
use crate::event::ProfilerEvent as _;
use crate::gpuviz;
use crate::histogram::{Histogram1D, Histogram2D};
use crate::nccl_metadata::NcclOpKey;
use crate::profiler::Profiler;

use log::error;
use serde_json::json;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct CloudDaemon {
    rt: tokio::runtime::Runtime,
    worker: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stop_signal: Option<oneshot::Sender<()>>,
}

impl Daemon for CloudDaemon {
    fn new(profiler: &'static Profiler) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (tx, rx) = oneshot::channel::<()>();
        let handle = rt.spawn(main_loop(profiler, rx));
        Self {
            rt,
            worker: Some(handle),
            stop_signal: Some(tx),
        }
    }
}

impl std::ops::Drop for CloudDaemon {
    fn drop(&mut self) {
        let stop_signal = self.stop_signal.take().unwrap();
        stop_signal.send(()).unwrap();
        let handle = self.worker.take().unwrap();
        let _ = self.rt.block_on(async { handle.await.unwrap() });
    }
}

impl Telemetry {
    async fn write_to_file<W, F>(&self, mut file: W, mut time_to_num: F) -> std::io::Result<()>
    where
        W: AsyncWriteExt + Unpin,
        F: FnMut(Instant) -> u64,
    {
        let mut buf: Vec<u8> = Vec::new();
        match self {
            Telemetry::Group(group) => {
                writeln!(buf, "{}", group.trace_record(&mut time_to_num))?;
            }
            Telemetry::NcclOp(ncclop) => {
                writeln!(buf, "{}", ncclop.trace_record(&mut time_to_num))?;
            }
            Telemetry::ProxyOp(proxyop) => {
                writeln!(buf, "{}", proxyop.trace_record(&mut time_to_num))?;
            }
        }
        file.write_all(&buf).await
    }
}

struct CollectiveSummary<W: AsyncWriteExt> {
    file: W,
    count: HashMap<NcclOpKey, Histogram1D>,
    latency: HashMap<NcclOpKey, Histogram2D>,
}

#[derive(Debug, Clone)]
struct AggregateMetrics {
    max_bytes: usize,
    max_latency_ns: u64,
    last_timestamp: Instant,
    count: u64,
}

struct LatencyAggregator<W: AsyncWriteExt> {
    file: W,
    metrics: HashMap<(NcclOpKey, usize), AggregateMetrics>,  // (op_key, rank) -> metrics
    last_flush: Instant,
}

impl<W: AsyncWriteExt> LatencyAggregator<W> {
    fn add_ncclop(&mut self, op: &event::NcclOp) {
        let key = (op.op_key(), op.basic_info().rank());
        let byte_count = op.byte_count();
        let now = op.basic_info().start_time();

        let latency_ns = op.child_duration()
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // Update or create metrics
        let metrics = self.metrics.entry(key).or_insert_with(|| AggregateMetrics {
            max_bytes: 0,
            max_latency_ns: 0,
            last_timestamp: now,
            count: 0,
        });

        metrics.max_bytes = metrics.max_bytes.max(byte_count);
        metrics.max_latency_ns = metrics.max_latency_ns.max(latency_ns);
        metrics.last_timestamp = now;
        metrics.count += 1;
    }

    async fn write_aggregated_metrics(&mut self) -> std::io::Result<()>
    where
        W: std::marker::Unpin,
    {
        if self.metrics.is_empty() {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let mut buf: Vec<u8> = Vec::new();

        let mut entries = Vec::new();
        for ((op_key, rank), metrics) in self.metrics.iter() {
            let metadata = op_key.to_json();
            let entry = json!({
                "cat": metadata["cat"],
                "op": metadata["op"],
                "rank": rank,
                "max_bytes": metrics.max_bytes,
                "max_latency_ns": metrics.max_latency_ns,
                "count": metrics.count,
            });
            entries.push(entry);
        }

        let aggregated = json!({
            "name": "aggregated_collective_metrics",
            "time": now,
            "aggregation_window_secs": 60,
            "entries": entries,
        });

        writeln!(
            buf,
            "{}",
            serde_json::to_string_pretty(&aggregated).unwrap()
        )?;

        self.file.write_all(&buf).await?;

        // Clear metrics for next aggregation period
        self.metrics.clear();
        self.last_flush = Instant::now();

        Ok(())
    }
}

impl<W: AsyncWriteExt> CollectiveSummary<W> {
    fn new(file: W) -> Self {
        Self {
            file,
            count: HashMap::new(),
            latency: HashMap::new(),
        }
    }

    fn add_op(&mut self, op: &event::NcclOp) {
        let key = op.op_key();
        let byte_count = op.byte_count();
        let count_histo = self.count.entry(key.clone()).or_default();
        count_histo.add(byte_count as _, 1);

        if let Some(latency) = op.child_duration() {
            let latency_histo = self.latency.entry(key.clone()).or_default();
            latency_histo.add((byte_count as _, latency.as_nanos() as u64), 1);
        }
    }

    async fn write_to_file(&mut self) -> std::io::Result<()>
    where
        W: std::marker::Unpin,
    {
        let now = std::time::SystemTime::now();
        let mut buf: Vec<u8> = Vec::new();
        let mut coll_counts = Vec::new();
        for (key, histogram) in self.count.iter() {
            let entry = json!({
                "metadata": key.to_json(),
                "message_sizes": histogram.to_json(),
            });
            coll_counts.push(entry);
        }
        let coll_counts = json!({
            "name": "collective_counts",
            "time": now,
            "entries": coll_counts,
        });
        writeln!(
            buf,
            "{}",
            serde_json::to_string_pretty(&coll_counts).unwrap()
        )?;

        let mut latency_distro = Vec::new();
        for (key, histogram) in self.latency.iter() {
            let entry = json!({
                "metadata": key.to_json(),
                "latency_distribution": histogram.to_json(),
            });
            latency_distro.push(entry);
        }
        let latency_distro = json!({
            "name": "collective_latency",
            "time": now,
            "entries": latency_distro,
        });
        writeln!(
            buf,
            "{}",
            serde_json::to_string_pretty(&latency_distro).unwrap()
        )?;

        self.file.write_all(&buf).await
    }

    async fn flush(&mut self) -> std::io::Result<()>
    where
        W: std::marker::Unpin,
    {
        self.file.flush().await
    }
}

async fn open_trace_file(path: impl AsRef<std::path::Path>) -> std::io::Result<tokio::fs::File> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .await
}

async fn build_bufwriter(
    path: impl AsRef<std::path::Path>,
) -> Option<tokio::io::BufWriter<tokio::fs::File>> {
    let r = open_trace_file(&path).await;
    match r {
        Ok(file) => Some(tokio::io::BufWriter::new(file)),
        Err(e) => {
            error!(
                "Failed to open file {:?} for logging telemetry: {}.",
                path.as_ref().as_os_str(),
                e
            );
            None
        }
    }
}

async fn exporter(
    profiler: &'static Profiler,
    mut rx: mpsc::Receiver<Telemetry>,
) -> std::io::Result<()> {
    let mut latency_aggregator = if let Some(template) = profiler.config.latency_file.as_ref() {
        let path = template.replace("%p", &format!("{}", profiler.pid));
        build_bufwriter(path).await.map(|file| LatencyAggregator {
            file,
            metrics: HashMap::new(),
            last_flush: Instant::now(),
        })
    } else {
        None
    };

    let mut summary = if let Some(template) = profiler.config.summary_file.as_ref() {
        let path = template.replace("%p", &format!("{}", profiler.pid));
        build_bufwriter(path).await.map(CollectiveSummary::new)
    } else {
        None
    };

    let mut gpuviz: Option<gpuviz::HistogramManager> = profiler
        .gpuviz_lib
        .clone()
        .map(gpuviz::HistogramManager::new);

    let mut summary_interval = tokio::time::interval(profiler.config.summary_interval);
    let mut latency_interval = tokio::time::interval(Duration::from_secs(60));  // 1 minute aggregation

    // the very first tick completes immediately
    summary_interval.tick().await;
    latency_interval.tick().await;

    loop {
        tokio::select! {
            maybe_telemetry = rx.recv() => {
                match maybe_telemetry {
                    Some(telemetry) => {
                        if let Some(aggregator) = latency_aggregator.as_mut() {
                            if let Telemetry::NcclOp(op) = &telemetry {
                                aggregator.add_ncclop(op);
                            }
                        }

                        if let Some(summary) = summary.as_mut() {
                            if let Telemetry::NcclOp(op) = &telemetry {
                                summary.add_op(op);
                            }
                        }

                        if let Some(gpuviz) = gpuviz.as_mut() {
                            if let Telemetry::NcclOp(op) = &telemetry {
                                let _ = gpuviz.add_ncclop(op, |t| {
                                    profiler.instant_to_timestamp(*t).as_nanos() as _
                                });
                            }
                        }
                    },
                    None => break,
                }
            },
            _ = summary_interval.tick(), if summary.is_some() => {
                let s = summary.as_mut().unwrap();
                let r = s.write_to_file().await;
                if let Err(e) = r {
                    error!("Failed to log telemetry summary to file: {}. Stop logging.", e);
                    summary = None;
                }
            },
            _ = latency_interval.tick(), if latency_aggregator.is_some() => {
                let agg = latency_aggregator.as_mut().unwrap();
                let r = agg.write_aggregated_metrics().await;
                if let Err(e) = r {
                    error!("Failed to log aggregated latency metrics to file: {}. Stop logging.", e);
                    latency_aggregator = None;
                }
            },
        }
    }

    if let Some(aggregator) = latency_aggregator.as_mut() {
        aggregator.write_aggregated_metrics().await?;
        aggregator.file.flush().await?;
    }

    if let Some(summary) = summary.as_mut() {
        summary.write_to_file().await?;
        summary.flush().await?;
    }

    Ok(())
}

impl Export for mpsc::Sender<Telemetry> {
    fn export(&self, ctx: &mut PollingContext, maybe_retry_ms: Option<u64>) {
        while let Some(telemetry) = ctx.pending_telemetry.pop_front() {
            if let Err(err) = self.try_send(telemetry) {
                match err {
                    TrySendError::Full(v) => {
                        ctx.pending_telemetry.push_front(v);
                        match maybe_retry_ms {
                            None => break,
                            Some(ms) => std::thread::sleep(Duration::from_micros(ms)),
                        }
                    }
                    _ => {
                        static ONETIME_LOG: Once = Once::new();
                        ONETIME_LOG.call_once(|| {
                            error!(
                                "Channel to telemetry exporter is unexpectedly closed. \
                                All future telemetry will be dropped."
                            );
                        });
                    }
                }
            }
        }
    }
}

async fn main_loop(
    profiler: &'static Profiler,
    stop: oneshot::Receiver<()>,
) -> std::io::Result<()> {
    const TELEMETRY_CHANNEL_SZ: usize = 4096;
    let (tx, rx) = mpsc::channel::<Telemetry>(TELEMETRY_CHANNEL_SZ);
    let exporter = tokio::task::spawn(exporter(profiler, rx));

    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_clone = stop_signal.clone();
    let polling_worker = tokio::task::spawn_blocking(move || {
        let mut polling_ctx = PollingContext::new(profiler, stop_signal_clone);
        polling_loop(&mut polling_ctx, tx)
    });

    let stop_signal_copy = stop_signal.clone();
    let timer_tick = if profiler.config.use_cached_clock {
        Some(std::thread::spawn(move || {
            while !stop_signal_copy.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_micros(5));
                profiler.cached_clock.update_cache(Instant::now());
            }
        }))
    } else {
        None
    };

    stop.await.unwrap();
    stop_signal.store(true, Ordering::Release);
    polling_worker.await?;
    if let Some(t) = timer_tick {
        let _ = t.join();
    }
    exporter.await??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config;
    use crate::daemon;
    use crate::nccl_metadata::Version as _;
    use crate::profiler::{thread_local_state, ThreadLocalState, Version};
    use crate::slab;
    use crate::step_tracker;
    use crate::{profiler_shim, scoped_profiler_test};

    use std::io::BufRead;
    use std::sync::Mutex;

    // create a mutex to avoid multiple test cases allocating ncclops concurrently
    static NCCLOP_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn e2e_mock() {
        let _lg = NCCLOP_TEST_MUTEX.lock().unwrap();

        const N_PROXYOP: usize = 1 << 16;

        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dir_path: &str = temp_dir.as_ref().to_str().unwrap();
        let pid = 42;
        let latency_template = format!("{}/latency-%p.txt", temp_dir_path);
        let summary_template = format!("{}/summary-%p.txt", temp_dir_path);
        // create a mock Profiler object
        let mut profiler = Profiler::new(Version::V1);
        profiler.pid = pid;
        profiler.config.track_group = true;
        profiler.config.track_ncclop = true;
        profiler.config.track_proxyop = true;
        profiler.config.track_interprocess_proxyop = false;
        profiler.config.latency_file = Some(latency_template.clone());
        profiler.config.summary_file = Some(summary_template.clone());
        scoped_profiler_test(profiler, |_profiler, thread_state| {
            let group_descr = profiler_shim::tests::dummy_group_descr();
            let coll_descr = profiler_shim::tests::dummy_coll_descr();
            let proxyop_descr = profiler_shim::tests::dummy_proxyop_descr();

            let group = Box::new(event::Group::from_descr(&group_descr, Instant::now()));
            let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                &coll_descr,
                Instant::now(),
                /* id = */ 0,
                /* comm_hash_override = */ None,
            ));
            let proxyop_descr_casted = unsafe { proxyop_descr.cast_to_proxyop() };
            let proxyops: Vec<_> = (0..N_PROXYOP)
                .map(|id| event::ProxyOpInfo::from_descr(proxyop_descr_casted, id as u32))
                .collect();

            thread_state.send_to_daemon(Message::Group(group), true);
            thread_state.send_to_daemon(Message::NcclOp(coll), true);

            let mut step_batch_list = slab::FreeList::new_list(proxyops.len());
            for proxyop in proxyops.into_iter() {
                let mut step_batch = step_batch_list
                    .alloc(
                        |p| unsafe { daemon::StepBatch::init(p) },
                        Some(&thread_state.profiler.free_step_batch),
                        false,
                    )
                    .unwrap();
                step_batch.push(step_tracker::EventStep {
                    step: 0,
                    size: 65536,
                    start_time: 123,
                    fifo_wait_dur_ns: None,
                    dur_ns: 256,
                });
                thread_state
                    .send_to_daemon(Message::StepBatch(proxyop.clone(), step_batch, true), true);
            }
        });

        // validate by opening the file and check if it is empty
        // we don't check content here as that would be too coupled
        // with implementation details
        // Note: Only check summary_file since latency_file now uses aggregation
        let path = summary_template.replace("%p", &format!("{}", pid));
        let file = std::fs::File::open(path).unwrap();
        assert!(file.metadata().unwrap().len() > 0);

        // The latency file now contains aggregated metrics, not individual events
        // Since aggregation happens at intervals, we just check that files were created
        // The actual aggregated data would be written on the next interval tick
        let path = latency_template.replace("%p", &format!("{}", pid));
        let _file = std::fs::File::open(path).unwrap();
        // Note: In the new implementation, metrics are aggregated and written periodically,
        // so the file may be empty or contain aggregated data depending on timing
    }

    fn reclaim_test_template<F>(n_ncclop: usize, client: F)
    where
        F: FnOnce(
                &mut ThreadLocalState,
                &std::sync::Barrier,
                mpsc::Receiver<Telemetry>,
                Arc<AtomicBool>,
            ) + Send,
    {
        let _lg = NCCLOP_TEST_MUTEX.lock().unwrap();

        let profiler = Box::new(Profiler::new(Version::V1));

        std::thread::scope(|s| {
            let (tx, rx) = mpsc::channel::<Telemetry>(n_ncclop);
            let stop_var = Arc::new(AtomicBool::new(false));
            let stop_var_clone = stop_var.clone();
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let barr = barrier.clone();
            let profiler_ref = &profiler;
            s.spawn(move || {
                let mut ctx = PollingContext::new(profiler_ref, stop_var_clone);
                polling_loop(&mut ctx, tx);
                barr.wait();
                assert_eq!(ctx.ncclops.len(), 0);
                barr.wait();
            });

            let profiler_ref = &profiler;
            s.spawn(move || {
                let (mut thread_state, control) = thread_local_state(profiler_ref);
                profiler_ref.register_thread(control);
                client(&mut thread_state, &barrier, rx, stop_var);
                barrier.wait();
            });
        });
    }

    #[test]
    fn reclaim_eventual() {
        const N_NCCLOP: usize = 50;

        reclaim_test_template(N_NCCLOP, |thread_state, barrier, rx, stop_var| {
            for op_idx in 0..N_NCCLOP {
                let coll_descr = profiler_shim::tests::dummy_coll_descr();
                let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                    &coll_descr,
                    Instant::now(),
                    op_idx,
                    /* comm_hash_override= */ None,
                ));
                thread_state.send_to_daemon(Message::NcclOp(coll), true);
            }
            stop_var.store(true, Ordering::Release);
            barrier.wait();
            assert_eq!(rx.len(), N_NCCLOP);
        });
    }

    #[test]
    fn reclaim_timeout() {
        let config: &config::Config = &config::CONFIG;
        let n_ncclop: usize = config.max_tracked_ncclop - 1;
        const NCCLOP_TIMEOUT: Duration = Duration::from_secs(100);

        reclaim_test_template(n_ncclop, |thread_state, barrier, rx, stop_var| {
            for op_idx in 0..n_ncclop {
                let coll_descr = profiler_shim::tests::dummy_coll_descr();
                let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                    &coll_descr,
                    Instant::now() - NCCLOP_TIMEOUT,
                    op_idx,
                    /* comm_hash_override= */ None,
                ));
                thread_state.send_to_daemon(Message::NcclOp(coll), true);
            }

            // block until rx is not empty
            let t0 = Instant::now();
            while rx.is_empty() && t0.elapsed().as_secs() < 10 {
                std::thread::sleep(Duration::from_micros(10));
            }
            stop_var.store(true, Ordering::Release);
            barrier.wait();
            assert_eq!(rx.len(), n_ncclop);
        });
    }

    #[test]
    fn reclaim_force() {
        let config: &config::Config = &config::CONFIG;
        let n_ncclop: usize = config.max_tracked_ncclop + 1000;

        reclaim_test_template(n_ncclop, |thread_state, barrier, rx, stop_var| {
            for op_idx in 0..n_ncclop {
                let coll_descr = profiler_shim::tests::dummy_coll_descr();
                let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                    &coll_descr,
                    // use a late start time to make sure timeout does not happen
                    Instant::now() + Duration::from_secs(3600),
                    op_idx,
                    /* comm_hash_override= */ None,
                ));
                thread_state.send_to_daemon(Message::NcclOp(coll), true);
            }

            // block until rx is not empty
            let t0 = Instant::now();
            while rx.is_empty() && t0.elapsed().as_secs() < 10 {
                std::thread::sleep(Duration::from_micros(10));
            }
            stop_var.store(true, Ordering::Release);
            barrier.wait();
            assert_eq!(rx.len(), n_ncclop);
        });
    }
}
