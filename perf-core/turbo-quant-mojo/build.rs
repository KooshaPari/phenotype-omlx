// build.rs — look for `libturbo_quant_mojo.dylib` produced by `mojo build --emit shared-lib`.
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

    // Search for pre-built Mojo shared library in common locations.
    let candidates = [
        manifest_dir.join("libturbo_quant_mojo.dylib"),
        out_dir.join("libturbo_quant_mojo.dylib"),
        PathBuf::from("/usr/local/lib/libturbo_quant_mojo.dylib"),
        PathBuf::from("/opt/homebrew/lib/libturbo_quant_mojo.dylib"),
    ];

    if let Some(found) = candidates.iter().find(|p| p.exists()) {
        let parent = found.parent().unwrap();
        // Normalize prebuilt Mojo artifacts that were emitted with an absolute
        // temporary install name; otherwise dyld ignores the consumer rpath.
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("install_name_tool")
                .args(["-id", "@rpath/libturbo_quant_mojo.dylib"])
                .arg(found)
                .status();
        }
        println!("cargo:rustc-link-search=native={}", parent.display());
        println!("cargo:rustc-link-lib=dylib=turbo_quant_mojo");
        // Tests and downstream binaries must resolve the colocated Mojo ABI at runtime.
        // Keep this explicit and local rather than requiring a machine-global DYLD path.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", parent.display());

        // Copy dylib to target/debug and target/debug/deps so downstream test binaries find it.
        if let Ok(target_dir) = env::var("OUT_DIR") {
            let target_path = PathBuf::from(target_dir);
            if let Some(debug_dir) = target_path.ancestors().nth(3) {
                let _ = std::fs::copy(found, debug_dir.join("libturbo_quant_mojo.dylib"));
                let _ = std::fs::copy(
                    found,
                    debug_dir.join("deps").join("libturbo_quant_mojo.dylib"),
                );
            }
        }
        println!("cargo:info=mojo staticlib found at {}", found.display());
    } else {
        println!(
            "cargo:warning=libturbo_quant_mojo.dylib not found — turbo-quant-mojo is a no-op stub"
        );
        println!("cargo:warning=build with:  mojo build mojo-src/turbo_quant.mojo --emit shared-lib -o libturbo_quant_mojo.dylib");
        println!("cargo:warning=install:     modular install mojo");
    }
}
