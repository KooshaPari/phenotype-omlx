.PHONY: check test clippy fmt all clean

CARGO_WORKSPACE_EXCLUDES := --exclude turbo-quant-mojo --exclude concurrent-exec-cuda

check:
	cargo check --workspace $(CARGO_WORKSPACE_EXCLUDES)

test:
	cargo test --workspace $(CARGO_WORKSPACE_EXCLUDES)

clippy:
	cargo clippy --workspace --all-targets $(CARGO_WORKSPACE_EXCLUDES) -- -D warnings

fmt:
	cargo fmt --all -- --check

all: check test clippy fmt

clean:
	cargo clean
