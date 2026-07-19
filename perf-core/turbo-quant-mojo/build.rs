// build.rs — compile Mojo shared library when `--features mojo` is enabled.
//
// Without the feature the crate is a Rust-only stub. With the feature, a missing
// Mojo SDK or failed `mojo build` is a hard error (fail loudly).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mojo_src = manifest_dir.join("mojo-src").join("turbo_quant.mojo");

    println!("cargo:rerun-if-changed={}", mojo_src.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MOJO_PATH");

    let feature_mojo = env::var("CARGO_FEATURE_MOJO").is_ok();
    if !feature_mojo {
        println!("cargo:warning=turbo-quant-mojo built without `mojo` feature — stub only");
        return;
    }

    let lib_name = if cfg!(target_os = "macos") {
        "libturbo_quant_mojo.dylib"
    } else if cfg!(target_os = "windows") {
        "turbo_quant_mojo.dll"
    } else {
        "libturbo_quant_mojo.so"
    };
    let lib_path = manifest_dir.join(lib_name);

    let mojo = env::var("MOJO_PATH")
        .ok()
        .map(PathBuf::from)
        .or_else(|| which("mojo"))
        .unwrap_or_else(|| {
            panic!(
                "turbo-quant-mojo: `--features mojo` requires the Mojo SDK in PATH \
                 (install: modular install mojo; or set MOJO_PATH)"
            );
        });

    let status = Command::new(&mojo)
        .args([
            "build",
            mojo_src.to_str().expect("mojo source path"),
            "-o",
            lib_path.to_str().expect("mojo output path"),
            "--emit",
            "shared-lib",
        ])
        .status()
        .unwrap_or_else(|e| panic!("turbo-quant-mojo: failed to invoke mojo build: {e}"));

    if !status.success() || !lib_path.exists() {
        panic!(
            "turbo-quant-mojo: `mojo build --emit shared-lib` failed — \
             cannot link native kernel with --features mojo"
        );
    }

    println!(
        "cargo:rustc-link-search=native={}",
        manifest_dir.display()
    );
    println!("cargo:rustc-link-lib=dylib=turbo_quant_mojo");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        manifest_dir.display()
    );
    println!("cargo:info=mojo shared library built at {}", lib_path.display());
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
