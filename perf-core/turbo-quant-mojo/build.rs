// build.rs — look for `libturbo_quant_mojo.a` produced by `mojo build`.
//
// If found (and the `mojo` feature is enabled), link it. Otherwise emit
// a warning and the crate compiles as a no-op stub.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mojo_src = manifest_dir.join("mojo-src").join("turbo_quant.mojo");

    println!("cargo:rerun-if-changed={}", mojo_src.display());
    println!("cargo:rerun-if-changed=build.rs");

    // Gate the native link behind the `mojo` feature. Without it the crate
    // builds as a Rust-only stub that returns graceful no-ops.
    let feature_mojo = env::var("CARGO_FEATURE_MOJO").is_ok();
    if !feature_mojo {
        println!("cargo:warning=turbo-quant-mojo built without `mojo` feature — stub only");
        return;
    }

    // Search for pre-built Mojo staticlib in common locations
    let candidates = [
        manifest_dir.join("libturbo_quant_mojo.a"),
        out_dir.join("libturbo_quant_mojo.a"),
        PathBuf::from("/usr/local/lib/libturbo_quant_mojo.a"),
        PathBuf::from("/opt/homebrew/lib/libturbo_quant_mojo.a"),
    ];

    if let Some(found) = candidates.iter().find(|p| p.exists()) {
        let parent = found.parent().unwrap();
        println!("cargo:rustc-link-search=native={}", parent.display());
        println!("cargo:rustc-link-lib=static=turbo_quant_mojo");
        println!("cargo:info=mojo staticlib found at {}", found.display());
    } else {
        println!("cargo:warning=libturbo_quant_mojo.a not found — turbo-quant-mojo is a no-op stub");
        println!("cargo:warning=build with:  mojo build mojo-src/turbo_quant.mojo -o libturbo_quant_mojo.a");
        println!("cargo:warning=install:     modular install mojo");
    }
}