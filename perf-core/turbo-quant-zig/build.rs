// build.rs — invoke `zig build-lib` to produce the C-ABI static library
// that the Rust wrapper links against. Only emits `rustc-link-*` directives
// when the .a file was actually produced, so the crate compiles cleanly
// on hosts that don't have the `zig` toolchain installed.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_src = manifest_dir.join("zig-src");
    let zig_file = zig_src.join("turbo_quant.zig");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lib_path = out_dir.join("libturbo_quant_zig.a");

    println!("cargo:rerun-if-changed={}", zig_file.display());
    println!("cargo:rerun-if-changed={}", manifest_dir.join("build.zig").display());
    println!("cargo:rerun-if-changed={}", manifest_dir.join("build.rs").display());
    println!("cargo:rerun-if-env-changed=ZIG_PATH");

    let zig = env::var("ZIG_PATH").ok()
        .map(PathBuf::from)
        .or_else(|| which("zig"));

    let zig = match zig {
        Some(z) => z,
        None => {
            // No zig toolchain available — skip the link entirely so cargo
            // doesn't try to bind against a missing .a. The crate's Rust
            // surface still compiles, but the native kernels will be no-ops
            // (handled via the feature-gated code in src/lib.rs).
            println!("cargo:warning=zig compiler not found in PATH; turbo-quant-zig will not be functional");
            println!("cargo:warning=install with: brew install zig  (or set ZIG_PATH)");
            return;
        }
    };

    let status = Command::new(&zig)
        .arg("build")
        .arg("-Doptimize=ReleaseFast")
        .current_dir(&manifest_dir)
        .status();

    let produced = match status {
        Ok(s) if s.success() => {
            let built = manifest_dir.join("zig-out/lib/libturbo_quant_zig.a");
            match std::fs::copy(&built, &lib_path) {
                Ok(_) => {
                    if let Some(ar) = which("llvm-ar").or_else(|| which("ar")) {
                        let _ = Command::new(&ar).args(["x", lib_path.to_str().unwrap()]).current_dir(&out_dir).status();
                        let object = out_dir.join("libturbo_quant_zig_zcu.o");
                        if object.exists() {
                            let _ = std::fs::remove_file(&lib_path);
                            let _ = Command::new(&ar)
                                .args(["rcs", lib_path.to_str().unwrap(), object.to_str().unwrap()])
                                .current_dir(&out_dir)
                                .status();
                        }
                    }
                    println!("cargo:info=zig build succeeded -> {}", lib_path.display());
                    true
                }
                Err(error) => {
                    println!("cargo:warning=failed to copy Zig archive: {error}");
                    false
                }
            }
        }
        Ok(s) => {
            println!("cargo:warning=zig build-lib failed with exit code: {:?}", s.code());
            false
        }
        Err(e) => {
            println!("cargo:warning=failed to invoke zig: {}", e);
            false
        }
    };

    if produced {
        // Only emit link directives when the .a was actually built
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=turbo_quant_zig");
        println!("cargo:info=linking against libturbo_quant_zig.a");
    } else {
        println!("cargo:warning=libturbo_quant_zig.a not produced; native bindings will be inert");
    }
}

fn which(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}
