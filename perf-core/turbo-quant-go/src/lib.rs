// turbo-quant-go — Rust workspace member for Go cgo binding.
// The Go source in go-src/ links against turbo-quant-c; `cargo test` here
// builds the C staticlib and runs `go test` as an end-to-end ABI check.

pub fn placeholder() -> &'static str {
    "turbo-quant-go"
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
            "cargo build --release -p turbo-quant-c failed — C ABI prerequisite for Go cgo"
        );
    }

    fn require_go() -> PathBuf {
        let go = which("go").unwrap_or_else(|| {
            panic!("go not found in PATH — install Go to run turbo-quant-go e2e tests")
        });
        go
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
    fn go_placeholder() {
        assert_eq!(placeholder(), "turbo-quant-go");
    }

    #[test]
    fn go_cgo_roundtrip_against_c_abi() {
        build_turbo_quant_c_release();
        let go = require_go();
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let status = Command::new(go)
            .args(["test", "-v", "./go-src/..."])
            .current_dir(&crate_dir)
            .status()
            .expect("spawn go test");
        assert!(
            status.success(),
            "go test failed — Go cgo binding must round-trip against turbo-quant-c ABI"
        );
    }
}
