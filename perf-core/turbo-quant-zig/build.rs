// build.rs — compile the Zig kernel via `zig build-obj` and link it.
//
// Zig 0.16's `build-lib` emits misaligned macOS archives; we emit a Mach-O
// object and link it directly. If Zig is missing or compilation fails,
// we emit a warning and skip linking (the crate will be a no-op stub).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_src = manifest_dir.join("zig-src");
    let zig_file = zig_src.join("root.zig");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_path = out_dir.join("turbo_quant_zig.o");

    println!("cargo:rerun-if-changed={}", zig_file.display());
    println!(
        "cargo:rerun-if-changed={}",
        zig_src.join("turbo_quant.zig").display()
    );
    println!("cargo:rerun-if-env-changed=ZIG_PATH");
    println!("cargo:rerun-if-env-changed=TURBO_QUANT_ZIG_SKIP");

    // Allow skipping Zig compilation entirely
    if env::var("TURBO_QUANT_ZIG_SKIP").is_ok() {
        println!(
            "cargo:warning=turbo-quant-zig: skipping Zig compilation (TURBO_QUANT_ZIG_SKIP set)"
        );
        return;
    }

    let zig = env::var("ZIG_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| which("zig"));

    let zig = match zig {
        Some(z) => z,
        None => {
            println!("cargo:warning=turbo-quant-zig: Zig compiler not found in PATH — skipping native build");
            return;
        }
    };

    let status = Command::new(&zig)
        .arg("build-obj")
        .arg("-O")
        .arg("ReleaseFast")
        .arg("--name")
        .arg("turbo_quant_zig")
        .arg(format!("-femit-bin={}", obj_path.display()))
        .arg(&zig_file)
        .status();

    match status {
        Ok(s) if s.success() && obj_path.exists() => {
            println!("cargo:rustc-link-arg={}", obj_path.display());
            println!("cargo:info=linking Zig object at {}", obj_path.display());
        }
        Ok(s) => {
            println!(
                "cargo:warning=turbo-quant-zig: `zig build-obj` exited with {} — skipping native build",
                s
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=turbo-quant-zig: failed to invoke zig: {e} — skipping native build"
            );
        }
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
