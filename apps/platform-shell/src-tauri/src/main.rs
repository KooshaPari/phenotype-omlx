//! Thin native shell — loads bench-cockpit (or any hub URL). No inference kernels.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let _ = app;
            // Window URL comes from tauri.conf.json build.devUrl / frontendDist.
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("platform-shell failed to start");
}
