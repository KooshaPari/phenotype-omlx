// turbo-quant-nim — Rust workspace member for Nim static-link binding.
// `cargo test` builds turbo-quant-c and runs the Nim unittest harness.

pub fn placeholder() -> &'static str {
    "turbo-quant-nim"
}

#[cfg(test)]
mod tests {
    use super::placeholder;
    use std::path::PathBuf;
    use std::process::Command;

    fn perf_core_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    fn build_turbo_quant_c_release() {
        let status = Command::new("cargo")
            .args(["build", "--release", "-p", "turbo-quant-c"])
            .current_dir(perf_core_dir())
            .status()
            .expect("spawn cargo build --release -p turbo-quant-c");
        assert!(
            status.success(),
            "cargo build --release -p turbo-quant-c failed — C ABI prerequisite for Nim"
        );
    }

    fn require_nim() -> PathBuf {
        which("nim").unwrap_or_else(|| {
            panic!("nim not found in PATH — install Nim to run turbo-quant-nim e2e tests")
        })
    }

    fn which(name: &str) -> Option<PathBuf> {
        std::env::var_os("PATH").and_then(|path| {
            std::env::split_paths(&path).find_map(|dir| {
                let candidate = dir.join(name);
                if candidate.is_file() {
                    Some(candidate)
                } else {
                    None
                }
            })
        })
    }

    #[test]
    fn nim_placeholder() {
        assert_eq!(placeholder(), "turbo-quant-nim");
    }

    #[test]
    fn nim_static_link_roundtrip_against_c_abi() {
        build_turbo_quant_c_release();
        let nim = require_nim();
        let nim_src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("nim-src");
        let status = Command::new(&nim)
            .args(["c", "-r", "--path:.", "turboquant_test.nim"])
            .current_dir(&nim_src)
            .status()
            .expect("spawn nim c -r turboquant_test.nim");
        assert!(
            status.success(),
            "nim unittest failed — Nim binding must round-trip against turbo-quant-c ABI"
        );
    }
}
