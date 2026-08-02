// build.rs — look for `libturbo_quant_mojo.dylib` produced by `mojo build --emit shared-lib`.
//
// If found (and the `mojo` feature is enabled), link it. Otherwise emit
// a warning and the crate compiles as a no-op stub.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(mojo_native)");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mojo_src = manifest_dir.join("mojo-src").join("turbo_quant.mojo");

    println!("cargo:rerun-if-changed={}", mojo_src.display());
    println!("cargo:rerun-if-changed=build.rs");

    // Gate the native link behind the `mojo` feature. Without it the crate
    // builds as a Rust-only stub that returns graceful no-ops.
    let feature_mojo_native = env::var("CARGO_FEATURE_MOJO_NATIVE").is_ok();
    if !feature_mojo_native {
        println!("cargo:warning=turbo-quant-mojo built without `mojo-native` feature — stub only");
        return;
    }

    // Search for a pre-built Mojo shared library using the target's native
    // filename.  The old implementation only looked for `.dylib`, which
    // silently disabled the native path on Linux and Windows.  Worse, a
    // stale artifact could enable `mojo_native` while not carrying the ABI
    // expected by `native.rs`, producing unresolved `tq_mojo_*` symbols at
    // link time.  Keep discovery conservative: no artifact means the
    // fail-closed Rust stub remains active.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let shared_names: &[&str] = match target_os.as_str() {
        "macos" => &["libturbo_quant_mojo.dylib"],
        "linux" => &["libturbo_quant_mojo.so"],
        "windows" => &["turbo_quant_mojo.dll", "libturbo_quant_mojo.dll"],
        _ => &[],
    };
    let search_dirs = [
        manifest_dir.clone(),
        out_dir,
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/opt/homebrew/lib"),
    ];
    let candidates: Vec<PathBuf> = search_dirs
        .iter()
        .flat_map(|dir| shared_names.iter().map(move |name| dir.join(name)))
        .collect();

    // On Windows a DLL alone cannot satisfy the Rust linker.  Require the
    // matching import library before enabling the native cfg; otherwise the
    // package intentionally stays on its validated fail-closed path.
    let import_library = if target_os == "windows" {
        let import_names: &[&str] = if target_env == "msvc" {
            &["turbo_quant_mojo.lib", "libturbo_quant_mojo.lib"]
        } else {
            &["libturbo_quant_mojo.dll.a", "turbo_quant_mojo.dll.a"]
        };
        search_dirs
            .iter()
            .flat_map(|dir| import_names.iter().map(move |name| dir.join(name)))
            .find(|path| path.exists())
    } else {
        None
    };

    if let Some(found) = candidates
        .iter()
        .find(|p| p.exists() && (target_os != "windows" || import_library.is_some()))
    {
        println!("cargo:rustc-cfg=mojo_native");
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
        if let Some(import) = import_library.as_ref() {
            if let Some(import_parent) = import.parent() {
                if import_parent != parent {
                    println!("cargo:rustc-link-search=native={}", import_parent.display());
                }
            }
        }
        println!("cargo:rustc-link-lib=dylib=turbo_quant_mojo");
        // Tests and downstream binaries must resolve the colocated Mojo ABI at runtime.
        // Keep this explicit and local rather than requiring a machine-global DYLD path.
        if target_os != "windows" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", parent.display());
        }

        // Copy dylib to target/debug and target/debug/deps so downstream test binaries find it.
        if let Ok(target_dir) = env::var("OUT_DIR") {
            let target_path = PathBuf::from(target_dir);
            if let Some(debug_dir) = target_path.ancestors().nth(3) {
                let file_name = found.file_name().unwrap();
                let _ = std::fs::copy(found, debug_dir.join(file_name));
                let _ = std::fs::copy(found, debug_dir.join("deps").join(file_name));
                if let Some(import) = import_library.as_ref() {
                    if let Some(import_name) = import.file_name() {
                        let _ = std::fs::copy(import, debug_dir.join(import_name));
                        let _ = std::fs::copy(import, debug_dir.join("deps").join(import_name));
                    }
                }
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
