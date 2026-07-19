// build.rs — always compile the Zig kernel via `zig build-obj` and link it.
//
// Zig 0.16's `build-lib` emits misaligned macOS archives; we emit a Mach-O
// object and link it directly. Missing toolchain or failed build panics.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_file = manifest_dir.join("zig-src").join("turbo_quant.zig");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_path = out_dir.join("turbo_quant_zig.o");

    println!("cargo:rerun-if-changed={}", zig_file.display());
    println!("cargo:rerun-if-env-changed=ZIG_PATH");

    let zig = env::var("ZIG_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| which("zig"))
        .unwrap_or_else(|| {
            panic!(
                "turbo-quant-zig: Zig compiler required in PATH \
                 (install: brew install zig; or set ZIG_PATH)"
            );
        });

    let status = Command::new(&zig)
        .arg("build-obj")
        .arg("-O")
        .arg("ReleaseFast")
        .arg("-fno-entry")
        .arg(format!("-femit-bin={}", obj_path.display()))
        .arg(&zig_file)
        .status()
        .unwrap_or_else(|e| panic!("turbo-quant-zig: failed to invoke zig build-obj: {e}"));

    if !status.success() || !obj_path.exists() {
        panic!(
            "turbo-quant-zig: `zig build-obj` failed — \
             cannot link native kernel (install: brew install zig)"
        );
    }

    println!("cargo:rustc-link-arg={}", obj_path.display());
    println!(
        "cargo:info=linking Zig object at {}",
        obj_path.display()
    );
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
