// main.swift — SwiftUI @main entry point for PhenotypeOMLX.
//
// A minimal `App` that hands off to `RootView`. Kept tiny on purpose so the
// app shell logic stays in `Views/`. No top-level configuration beyond a
// single WindowGroup (single-window desktop tool).

import SwiftUI

@main
struct PhenotypeOMLXApp: App {
    var body: some Scene {
        WindowGroup("Phenotype oMLX") {
            RootView()
                .frame(minWidth: 900, minHeight: 600)
        }
        .windowResizability(.contentMinSize)
    }
}