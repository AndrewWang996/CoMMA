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

use std::env;
use std::path::PathBuf;

fn write_binding(hdr_path: &str, output_path: &std::path::Path) {
    let bindings = bindgen::Builder::default()
        .header(hdr_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .impl_debug(true)
        .generate()
        .unwrap_or_else(|e| panic!("Unable to generate bindings for {}: {}", hdr_path, e));
    bindings
        .write_to_file(output_path)
        .unwrap_or_else(|e| panic!("Could not write bindings to {:?}: {}!", output_path, e));
}

fn cmake_build(dir: &str, lib_name: &str, def_vars: &[(&str, &str)]) {
    let mut dst = cmake::Config::new(dir);
    for (k, v) in def_vars {
        dst.define(k, v);
    }
    let dst = dst.build();
    println!("cargo:rerun-if-changed={}", dir);
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static={}", lib_name);

    let bindings = bindgen::Builder::default()
        .header(format!("{}/c-helpers.h", dir))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .unwrap_or_else(|e| panic!("Unable to generate bindings for {}: {}", dir, e));

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join(format!("{}_shim_inner.rs", lib_name)))
        .expect("Could not write bindings!");
}

fn main() {
    let profiler_hdr_dir = "third_party/nccl/ext-profiler/example/nccl";
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    write_binding(
        &format!("{}/profiler.h", profiler_hdr_dir),
        &out_path.join("profiler_shim_inner.rs"),
    );

    println!("cargo:rerun-if-changed={}", profiler_hdr_dir);

    const GPUVIZ_HDR: &str = "GPUViz/src/nccl_stats.h";
    let gpuviz_shim = out_path.join("gpuviz_shim_inner.rs");
    let bindings = bindgen::Builder::default()
        .header(GPUVIZ_HDR)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .impl_debug(true)
        .allowlist_type("ncclStats.*")
        .generate()
        .unwrap_or_else(|e| panic!("Unable to generate bindings for {}: {}", GPUVIZ_HDR, e));

    let build_profile = env::var("PROFILE").unwrap_or(String::from("debug"));
    let cmake_build_type = match build_profile.as_str() {
        "release" => "Release",
        _ => "Debug",
    };

    cmake_build(
        "c-helpers",
        "c_helpers",
        &[("CMAKE_BUILD_TYPE", cmake_build_type)],
    );

    bindings
        .write_to_file(&gpuviz_shim)
        .unwrap_or_else(|e| panic!("Could not write bindings to {:?}: {}!", gpuviz_shim, e));
    println!("cargo:rerun-if-changed=GPUViz");
}
