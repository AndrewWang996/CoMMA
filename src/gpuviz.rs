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

use crate::config;
use crate::daemon;
use crate::event;
use crate::event::ProfilerEvent as _;
use crate::nccl_metadata;
use crate::nccl_metadata::NcclOpKey;
use crate::step_tracker::EventStep;

use log::{log, Level};
use std::collections::HashMap;
use std::ffi::CStr;
use std::mem::{zeroed, MaybeUninit};
use std::path::Path;
use std::sync::{Arc, Once};

pub mod shim {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(improper_ctypes)]
    #![allow(unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/gpuviz_shim_inner.rs"));
}

mod c_helpers {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(improper_ctypes)]
    #![allow(unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/c_helpers_shim_inner.rs"));
}

pub type ConnectionIdentifier = shim::ncclStatsConnectionIdentifier;
pub type Measurement = shim::ncclStatsOperationMetric;

pub const GPUVIZ_LIB_NAME: &str = "libGPUViz.so";
const GPUVIZ_ROOT_SYMBOL: &[u8] = b"nccl_telemetry_stats_plugin_v1\0";
const GPUVIZ_ROOT_SYMBOL_V2: &[u8] = b"nccl_telemetry_stats_plugin_v2\0";

#[derive(Debug)]
pub struct GpuViz {
    _lib_handle: libloading::Library,
    symbols: *mut shim::ncclStatsPlugin_t,
    stats_handle: libc::uintptr_t,
    epoch_dt: u64,
}

/// # Safety
///
/// ncclStatsPlugin_t only contains function pointers that are MT safe
unsafe impl Send for GpuViz {}
unsafe impl Sync for GpuViz {}

static LOGGER_INIT: Once = Once::new();

/// # Safety
///
/// This function is a callback to be passed to C library as
/// a logging callback.
/// file and msg should be valid pointers to C string
unsafe extern "C" fn inner_logger_fn(
    lvl: shim::ncclDebugLogLevel,
    _flags: u64,
    file: *const libc::c_char,
    line: libc::c_int,
    msg: *const libc::c_char,
    _len: libc::c_int,
) {
    let log_level = match lvl {
        shim::ncclDebugLogLevel_NCCL_LOG_TRACE => Level::Trace,
        shim::ncclDebugLogLevel_NCCL_LOG_VERSION => Level::Info,
        shim::ncclDebugLogLevel_NCCL_LOG_INFO => Level::Info,
        shim::ncclDebugLogLevel_NCCL_LOG_WARN => Level::Warn,
        shim::ncclDebugLogLevel_NCCL_LOG_ABORT => Level::Error,
        _ => Level::Trace,
    };
    let file_str = CStr::from_ptr(file).to_str().unwrap();
    let msg_str = CStr::from_ptr(msg).to_str().unwrap();
    log!(log_level, "GPUViz({}:{}) {}", file_str, line, msg_str);
}

impl GpuViz {
    pub fn from_path<P: AsRef<Path>>(path: P, epoch_dt: u64) -> std::io::Result<Self> {
        let path_str = path.as_ref().to_str().unwrap();
        let path_osstr = std::ffi::OsString::from(path_str);
        let convert_err = |err| std::io::Error::other(err);
        // SAFETY: the shared object we load here does not have init and termination
        // routines.
        unsafe {
            let lib = libloading::Library::new(&path_osstr).map_err(convert_err)?;
            let root_symbol: libloading::Symbol<*mut shim::ncclStatsPlugin_t> =
                lib.get(GPUVIZ_ROOT_SYMBOL).map_err(convert_err)?;

            let v2_symbol: Option<libloading::Symbol<*mut shim::ncclStatsPlugin_v2_t>> =
                lib.get(GPUVIZ_ROOT_SYMBOL_V2).ok();

            let stats_plugin = root_symbol.into_raw().as_raw_ptr().cast();

            let stats_handle = if let Some(v2_symbol) = v2_symbol {
                let v2_plugin = v2_symbol.into_raw().as_raw_ptr().cast();
                Self::init_v2(&*v2_plugin)?
            } else {
                Self::init(&*stats_plugin)?
            };
            let r = Self {
                _lib_handle: lib,
                symbols: stats_plugin,
                stats_handle,
                epoch_dt,
            };
            Ok(r)
        }
    }

    pub fn dylib(epoch_dt: u64) -> std::io::Result<Self> {
        Self::from_path(&config::CONFIG.gpuviz_lib, epoch_dt)
    }

    fn symbols(&self) -> &shim::ncclStatsPlugin_t {
        // SAFETY: self.symbols is not null upon construction
        unsafe { &*self.symbols }
    }

    fn init(plugin: &shim::ncclStatsPlugin_t) -> std::io::Result<libc::uintptr_t> {
        // SAFETY: calling FFI function (expected to be MT safe)
        unsafe {
            LOGGER_INIT.call_once(|| {
                c_helpers::init_logger(Some(inner_logger_fn));
            });
            let mut handle = MaybeUninit::uninit();
            let distribution_bitmap = shim::ncclStatsDistributionType_SendLatencySW
                | shim::ncclStatsDistributionType_RecvLatencySW
                | shim::ncclStatsDistributionType_SendMessageSize
                | shim::ncclStatsDistributionType_RecvMessageSize;
            let r = plugin.init.unwrap()(
                Some(c_helpers::logger_helper),
                distribution_bitmap as _,
                handle.as_mut_ptr(),
            );
            if r != shim::ncclResult_t_ncclSuccess {
                return Err(std::io::Error::other("gpuviz error"));
            }
            Ok(handle.assume_init())
        }
    }

    fn init_v2(plugin: &shim::ncclStatsPlugin_v2_t) -> std::io::Result<libc::uintptr_t> {
        // SAFETY: calling FFI function (expected to be MT safe)
        unsafe {
            LOGGER_INIT.call_once(|| {
                c_helpers::init_logger(Some(inner_logger_fn));
            });
            let mut handle = MaybeUninit::uninit();
            let distribution_bitmap = shim::ncclStatsDistributionType_SendLatencySW
                | shim::ncclStatsDistributionType_RecvLatencySW
                | shim::ncclStatsDistributionType_SendMessageSize
                | shim::ncclStatsDistributionType_RecvMessageSize;
            let r = plugin.init.unwrap()(
                Some(c_helpers::logger_helper),
                distribution_bitmap as _,
                config::CONFIG.telemetry_mode as _,
                c"Profiler".as_ptr(),
                handle.as_mut_ptr(),
            );
            if r != shim::ncclResult_t_ncclSuccess {
                return Err(std::io::Error::other("gpuviz error"));
            }
            Ok(handle.assume_init())
        }
    }

    pub fn add_connection(
        self: &Arc<Self>,
        conn_id: &ConnectionIdentifier,
        is_send: bool,
        conn_type: ConnectionType,
    ) -> std::io::Result<Connection> {
        // SAFETY: calling FFI function (expected to be MT safe)
        let conn_handle = unsafe {
            let mut handle = MaybeUninit::uninit();
            let r = self.symbols().addConnection.unwrap()(
                self.stats_handle,
                conn_id as *const _,
                handle.as_mut_ptr(),
            );
            if r != shim::ncclResult_t_ncclSuccess {
                return Err(std::io::Error::other("gpuviz addConnection error"));
            }
            handle.assume_init()
        };
        Ok(Connection {
            handle: conn_handle,
            is_send,
            conn_type,
            gpuviz: self.clone(),
            notify_measurement: self.symbols().notifyOperationMeasurement.unwrap(),
        })
    }
}

impl std::ops::Drop for GpuViz {
    fn drop(&mut self) {
        // SAFETY: calling FFI function (expected to be MT safe)
        // needed to stop infinite retry
        unsafe {
            let r = self.symbols().destroy.unwrap()(self.stats_handle);
            debug_assert_eq!(r, shim::ncclResult_t_ncclSuccess);
        }
    }
}

type NotifyFn = unsafe extern "C" fn(
    statsConnectionHandle: usize,
    measurement: *const shim::ncclStatsOperationMetric,
) -> shim::ncclResult_t;

#[derive(Debug)]
pub struct Connection {
    handle: libc::uintptr_t,
    is_send: bool,
    conn_type: ConnectionType,
    gpuviz: Arc<GpuViz>,
    notify_measurement: NotifyFn,
}

impl AsRef<Connection> for Connection {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Connection {
    pub fn notify_measurement(&self, measurement: &Measurement) -> std::io::Result<()> {
        // SAFETY: calling FFI function (expected to be MT safe)
        unsafe {
            // let api = &self.gpuviz.symbols().notifyOperationMeasurement.unwrap();
            let api = self.notify_measurement;
            let r = api(self.handle, measurement as *const _);
            if r == shim::ncclResult_t_ncclSuccess {
                Ok(())
            } else {
                Err(std::io::Error::other("gpuviz notifyMeasurement error"))
            }
        }
    }

    pub fn notify_latency(
        &self,
        ts: u64,
        latency: u64,
        sz: usize,
        is_send: bool,
    ) -> std::io::Result<()> {
        // SAFETY: zero initialize or memcpy C struct
        let mut lat: shim::ncclStatsLatencyMeasurement = unsafe { zeroed() };
        lat.latency_type = shim::ncclStatsLatencyType_LatencySoftware;
        lat.latency_in_nanoseconds = latency;

        let mut m: Measurement = unsafe { zeroed() };
        m.type_ = if is_send {
            shim::ncclStatsOpType_OperationTypeChunkSend
        } else {
            shim::ncclStatsOpType_OperationTypeChunkRecv
        };
        m.collective_id = 0;
        m.op_id = 0;
        m.op_sz = sz as _;
        m.op_start_time = ts;
        m.num_measurements = 1;
        m.measurements = &lat as _;
        self.notify_measurement(&m)
    }
}

impl std::ops::Drop for Connection {
    fn drop(&mut self) {
        // SAFETY: calling FFI function (expected to be MT safe)
        unsafe {
            let close_type = shim::ncclStatsConnectionCloseType_ConnectionCloseRemoteTerminate;
            let reason: &CStr = c"Drop";
            let r = self.gpuviz.symbols().deleteConnection.unwrap()(
                self.handle,
                close_type,
                reason.as_ptr(),
            );
            debug_assert_eq!(r, shim::ncclResult_t_ncclSuccess);
        }
    }
}

impl daemon::AtomicHistogram<EventStep> for Connection {
    fn record(&self, step: &EventStep) {
        let ts = step.start_time + self.gpuviz.epoch_dt;
        let latency = match self.conn_type {
            ConnectionType::Net => step.dur_ns as _,
            ConnectionType::E2e => (step.dur_ns + step.fifo_wait_dur_ns.unwrap_or(0)) as _,
        };
        let _ = self.notify_latency(ts, latency, step.size, self.is_send);
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum ConnectionType {
    Net,
    E2e,
}

#[derive(Debug)]
pub struct HistogramManager<C: AsRef<Connection> + From<Connection> = Connection> {
    gpuviz: Arc<GpuViz>,
    connections: HashMap<(NcclOpKey, ConnectionType), C>,
}

impl<C: AsRef<Connection> + From<Connection>> HistogramManager<C> {
    pub fn new(lib: Arc<GpuViz>) -> Self {
        Self {
            gpuviz: lib,
            connections: HashMap::new(),
        }
    }

    fn new_connection(
        gpuviz: &Arc<GpuViz>,
        key: &(NcclOpKey, ConnectionType),
    ) -> std::io::Result<Connection> {
        // create a dummy connection for the entire profiler
        // SAFETY: zero initialize or memcpy C struct
        let mut conn_id: ConnectionIdentifier = unsafe { zeroed() };
        conn_id.nccl_plugin_name = c"profiler".as_ptr();
        conn_id.nccl_plugin_type = shim::ncclStatsPluginType_ProfilerPlugin;
        conn_id.conn_type = shim::ncclStatsConnectionType_EntityProfilerPluginConnection;
        let mut conn: shim::ncclStatsProfilerPluginConnection = unsafe { zeroed() };
        let mut sockaddr: libc::sockaddr_in = unsafe { zeroed() };
        sockaddr.sin_family = libc::AF_UNSPEC as _;
        sockaddr.sin_addr.s_addr = 0x2a2a2a2a;
        sockaddr.sin_port = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(
                &sockaddr as *const _ as *const u8,
                &mut conn.local_endpoint as *mut _ as *mut u8,
                std::mem::size_of_val(&sockaddr),
            );
        }
        let mut is_send = true;
        match &key.0 {
            NcclOpKey::Collective(comm, rank, op_type, algo, _proto) => {
                conn.local_comm_hash = *comm;
                conn.local_rank = *rank as _;
                conn.remote_or_root_rank = 0;
                conn.collective_type = op_type.name_cstr().as_ptr();
                conn.collective_algorithm = nccl_metadata::algo::name_cstr(*algo).as_ptr();
            }
            NcclOpKey::P2pSend(comm, local, peer) => {
                conn.local_comm_hash = *comm;
                conn.local_rank = *local as _;
                conn.remote_or_root_rank = *peer as _;
                conn.collective_type = c"nccl_send".as_ptr();
            }
            NcclOpKey::P2pRecv(comm, local, peer) => {
                conn.local_comm_hash = *comm;
                conn.local_rank = *local as _;
                conn.remote_or_root_rank = *peer as _;
                conn.collective_type = c"nccl_recv".as_ptr();
                is_send = false;
            }
            NcclOpKey::NetSend(comm, local, peer) => {
                conn.local_comm_hash = *comm;
                conn.local_rank = *local as _;
                conn.remote_or_root_rank = *peer as _;
                conn.collective_type = match key.1 {
                    ConnectionType::Net => c"net_send.network_send_latency".as_ptr(),
                    ConnectionType::E2e => c"net_send.issue_to_completion".as_ptr(),
                };
            }
            NcclOpKey::NetRecv(comm, local, peer) => {
                conn.local_comm_hash = *comm;
                conn.local_rank = *local as _;
                conn.remote_or_root_rank = *peer as _;
                conn.collective_type = c"net_recv".as_ptr();
                is_send = false;
            }
        }

        conn_id.connection.profiler_conn = conn;
        conn_id.conn_type = shim::ncclStatsConnectionType_EntityProfilerPluginConnection;

        gpuviz.add_connection(&conn_id, is_send, key.1)
    }

    pub fn get_connection(
        &mut self,
        key: &NcclOpKey,
        conn_type: Option<ConnectionType>,
    ) -> std::io::Result<&C> {
        let conn_type = conn_type.unwrap_or(ConnectionType::Net);
        let key = (key.clone(), conn_type);
        if !self.connections.contains_key(&key) {
            let new_conn = Self::new_connection(&self.gpuviz, &key)?;
            self.connections.insert(key.clone(), C::from(new_conn));
        }
        Ok(self.connections.get(&key).unwrap())
    }

    pub fn add_ncclop<F>(&mut self, op: &event::NcclOp, time_to_num: F) -> std::io::Result<()>
    where
        F: FnOnce(&std::time::Instant) -> u64,
    {
        let basic_info = op.basic_info();
        let start_time = op
            .child_start_time()
            .unwrap_or_else(|| basic_info.start_time());
        let end_time = basic_info.end_time().unwrap_or(start_time);
        let ts = time_to_num(&start_time);
        let latency = (end_time - start_time).as_nanos() as u64;
        let sz = op.byte_count();

        let key = op.op_key();
        let connection = self.get_connection(&key, None)?;
        // we report nccl op latency as "send"
        connection
            .as_ref()
            .notify_latency(ts, latency, sz, /* is_send = */ true)?;
        Ok(())
    }
}
