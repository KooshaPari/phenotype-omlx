use eval_harness::verifier::{verify_artifact, verify_self, VerifyOutcome};
use serde_json::Value;
use std::{env, fs, process::ExitCode};

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().collect();
    if args.len() == 1 {
        return match verify_self() {
            VerifyOutcome::Accept => ExitCode::SUCCESS,
            VerifyOutcome::Reject { message } | VerifyOutcome::InternalMismatch { message } => {
                eprintln!("self-test failed: {message}");
                ExitCode::from(2)
            }
        };
    }
    if args.len() != 2 {
        eprintln!("usage: verify_contract [evaluation-report.json]");
        return ExitCode::from(2);
    }
    let bytes = match fs::read(&args[1]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("internal error reading artifact: {error}");
            return ExitCode::from(2);
        }
    };
    let artifact: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("reject: invalid JSON: {error}");
            return ExitCode::from(1);
        }
    };
    match verify_artifact(&artifact) {
        VerifyOutcome::Accept => ExitCode::SUCCESS,
        VerifyOutcome::Reject { message } => {
            eprintln!("reject: {message}");
            ExitCode::from(1)
        }
        VerifyOutcome::InternalMismatch { message } => {
            eprintln!("internal mismatch: {message}");
            ExitCode::from(2)
        }
    }
}
