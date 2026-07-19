const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addStaticLibrary(.{
        .name = "turbo_quant_zig",
        .root_source_file = b.path("zig-src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);
}
