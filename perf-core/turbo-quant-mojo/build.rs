// build.rs — look for `libturbo_quant_mojo.dylib` produced by `mojo build --emit shared-lib`.
//
// Missing Mojo SDK or a failed `mojo build` is a hard error (fail loudly).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mojo_src = manifest_dir.join("mojo-src").join("turbo_quant.mojo");

    println!("cargo:rerun-if-changed={}", mojo_src.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MOJO_PATH");

    // Gate the native link behind the `mojo` feature. Without it the crate
    // builds as a Rust-only stub that returns graceful no-ops.
    let feature_mojo = env::var("CARGO_FEATURE_MOJO").is_ok();
    if !feature_mojo {
        println!("cargo:warning=turbo-quant-mojo built without `mojo` feature — stub only");
        return;
    }

    // Search for pre-built Mojo shared library in common locations.
    let candidates = [
        manifest_dir.join("libturbo_quant_mojo.dylib"),
        out_dir.join("libturbo_quant_mojo.dylib"),
        PathBuf::from("/usr/local/lib/libturbo_quant_mojo.dylib"),
        PathBuf::from("/opt/homebrew/lib/libturbo_quant_mojo.dylib"),
    ];

    if let Some(found) = candidates.iter().find(|p| p.exists()) {
        let parent = found.parent().unwrap();
        println!("cargo:rustc-link-search=native={}", parent.display());
        println!("cargo:rustc-link-lib=dylib=turbo_quant_mojo");
        println!("cargo:info=mojo staticlib found at {}", found.display());
    } else {
        println!("cargo:warning=libturbo_quant_mojo.dylib not found — turbo-quant-mojo is a no-op stub");
        println!("cargo:warning=build with:  mojo build mojo-src/turbo_quant.mojo --emit shared-lib -o libturbo_quant_mojo.dylib");
        println!("cargo:warning=install:     modular install mojo");
    }
}
