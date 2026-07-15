// swift-tools-version:5.9
// Package.swift — PhenotypeOMLX (macOS SwiftUI GUI)
//
// Native SwiftUI shell that wraps the Python `omlx-research` CLI via Process.
// Does NOT link MLX-C / FFI directly — keeps the build hermetic and avoids
// Python-version coupling (3.11 framework vs 3.12+ venv).

import PackageDescription

let package = Package(
    name: "PhenotypeOMLX",
    platforms: [
        .macOS(.v14) // Sonoma — required for NavigationSplitView refinements
    ],
    products: [
        .executable(name: "PhenotypeOMLX", targets: ["PhenotypeOMLX"])
    ],
    targets: [
        .executableTarget(
            name: "PhenotypeOMLX",
            path: "Sources/PhenotypeOMLX"
        )
    ]
)