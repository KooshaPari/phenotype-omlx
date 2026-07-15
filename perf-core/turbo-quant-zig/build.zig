// Build script for the turbo-quant Zig FFI bridge.
//
// This script:
//   1. Invokes `cargo build --release` against the sibling `turbo-quant` Rust
//      crate so that a C-ABI static library is available at link time.
//   2. Compiles the Zig source (`src/root.zig`) — which contains only
//      `extern fn` declarations for the Rust C-ABI symbols — into a dynamic
//      library that re-exports those symbols via the linker.
//
// On macOS the output is `libturbo_quant_zig.dylib`.
//
// IMPORTANT — Rust crate-type requirement:
//   The Rust crate (`perf-core/turbo-quant/Cargo.toml`) must declare its
//   `[lib]` section as a staticlib in order for cargo to emit a `.a` file
//   that this build can link against. Currently the Rust crate does NOT
//   declare `crate-type = ["staticlib"]`, nor does it expose
//   `#[no_mangle] pub extern "C" fn turbo_quant_encode/decode` symbols. See
//   `README.md` for the exact patch that needs to land on the Rust side
//   before this bridge can be linked end-to-end.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1) Build the Rust staticlib via cargo.
    //
    //    The step is gated behind a feature/target so we don't pay the
    //    cargo cost on plain `zig build` invocations from CI that don't need
    //    it; pass `-Denable-rust=true` to opt in.
    const enable_rust = b.option(bool, "enable-rust", "Also build the Rust staticlib via cargo before linking") orelse true;

    if (enable_rust) {
        const cargo_build = b.addSystemCommand(&.{
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            "../turbo-quant/Cargo.toml",
        });
        // Make the install step wait for cargo to finish so the .a is on disk
        // before the Zig linker runs.
        b.getInstallStep().dependOn(&cargo_build.step);
    }

    // 2) Compile the Zig source into a dynamic library and link the Rust
    //    staticlib into it. The Zig source contains only `extern fn`
    //    declarations, so the resulting .dylib exports the Rust symbols
    //    verbatim — Python ctypes can load it directly.
    const lib = b.addLibrary(.{
        .name = "turbo_quant_zig",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Link the Rust staticlib produced by `cargo build --release`.
    // The path is relative to this build.zig, i.e. `perf-core/turbo-quant-zig/`.
    lib.root_module.addObjectFile(b.path("../turbo-quant/target/release/libturbo_quant.a"));

    // The Rust crate links the C standard library via `half`/serde/etc., so
    // make sure libc is pulled in on the Zig side too.
    lib.root_module.link_libc = true;

    // Mark `turbo_quant_encode` / `turbo_quant_decode` as symbols that must
    // survive `--gc-sections` / dead-code elimination — without this, the
    // linker can drop them because no Zig code references them.
    lib.root_module.force_undefined_symbols.put(b.allocator, "turbo_quant_encode", {}) catch @panic("OOM");
    lib.root_module.force_undefined_symbols.put(b.allocator, "turbo_quant_decode", {}) catch @panic("OOM");

    b.installArtifact(lib);

    // A convenience `zig build check` step that just `zig build-exe`-style
    // type-checks root.zig without producing a library. Useful for CI.
    const check = b.step("check", "Type-check src/root.zig without linking");
    const check_lib = b.addLibrary(.{
        .name = "turbo_quant_zig_check",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // No object file in check mode — just want semantic analysis.
    check.dependOn(&check_lib.step);
}