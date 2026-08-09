//! AgilePlus Governance CLI
//!
//! Command-line interface for the governance system.

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::PathBuf;

use agileplus_governance::{
    GovernanceClient, PolicyCheck, PolicyContext, PromotionRequest, ReleaseChannel,
};

/// AgilePlus Governance CLI
#[derive(Parser)]
#[command(name = "agileplus-governance")]
#[command(about = "AgilePlus Governance System CLI")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Governance database path
    #[arg(short, long, default_value = ".agileplus/governance.db")]
    db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Check a policy
    Policy {
        /// Action to check
        #[arg(short, long)]
        action: String,

        /// Resource name
        #[arg(short, long)]
        resource: String,

        /// Current channel (alpha, canary, beta, rc, prod)
        #[arg(short, long)]
        channel: Option<String>,
    },

    /// Promote a release
    Promote {
        /// Crate name
        #[arg(short, long)]
        crate_name: String,

        /// From channel
        #[arg(short, long)]
        from: String,

        /// To channel
        #[arg(short, long)]
        to: String,

        /// Audit identity for the promotion request (defaults to AGILEPLUS_REQUESTED_BY or the OS username)
        #[arg(long)]
        requested_by: Option<String>,
    },

    /// Check connection to remote governance
    Status {},
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        std::env::set_var("RUST_LOG", "agileplus_governance=debug");
        tracing_subscriber::fmt::init();
    }

    match &cli.command {
        Commands::Policy {
            action,
            resource,
            channel,
        } => {
            let client = GovernanceClient::with_defaults().await?;
            let mut context = PolicyContext::new()
                .with_action(action)
                .with_resource(resource, None);

            if let Some(ch) = channel {
                match ch.parse::<ReleaseChannel>() {
                    Ok(channel) => context = context.with_channel(channel),
                    Err(e) => {
                        eprintln!("Invalid channel '{}': {}", ch, e);
                    }
                }
            }

            let check = PolicyCheck {
                resource: resource.clone(),
                action: action.clone(),
                context,
            };

            let result = client.check_policy(check).await?;
            print_policy_result(&result);
        }

        Commands::Promote {
            crate_name,
            from,
            to,
            requested_by,
        } => {
            let client = GovernanceClient::with_defaults().await?;

            let from_channel = match from.parse::<ReleaseChannel>() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Invalid from channel '{}': {}", from, e);
                    return Ok(());
                }
            };

            let to_channel = match to.parse::<ReleaseChannel>() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Invalid to channel '{}': {}", to, e);
                    return Ok(());
                }
            };

            let request = PromotionRequest {
                package: crate_name.clone(),
                from: from_channel,
                to: to_channel,
                requested_by: resolve_promotion_requester(
                    requested_by
                        .clone()
                        .or_else(|| std::env::var(PROMOTION_REQUESTER_ENV).ok()),
                    whoami::username(),
                )?,
                version: "0.1.0".to_string(),
                metadata: None,
            };

            let result = client.check_promotion(request).await?;
            if result.allowed {
                println!("✓ ALLOWED: {}", result.reason);
                if let Some(ref metadata) = result.channel_metadata {
                    println!("  Channel: {}", metadata.channel);
                    println!("  Version: {}", metadata.version);
                }
                if !result.warnings.is_empty() {
                    println!("  Warnings:");
                    for w in &result.warnings {
                        println!("    - {}", w);
                    }
                }
            } else {
                println!("✗ DENIED: {}", result.reason);
                if !result.policy_failures.is_empty() {
                    println!("  Policy failures:");
                    for f in &result.policy_failures {
                        println!("    - {}", f);
                    }
                }
            }
        }

        Commands::Status {} => {
            let client = GovernanceClient::with_defaults().await?;
            let status = client.status().await;
            println!("=== Governance Status ===");
            println!("Initialized: {}", status.initialized);
            println!("Remote enabled: {}", status.remote_enabled);
            println!("Local enabled: {}", status.local_enabled);
            println!("Connection: {:?}", status.connection_status);
            println!("Pending sync operations: {}", status.pending_operations);
            if let Some(last_sync) = status.last_sync {
                println!("Last sync: {}", last_sync);
            }
        }
    }

    Ok(())
}

fn print_policy_result(result: &agileplus_governance::PolicyResult) {
    if result.allowed {
        println!("✓ ALLOWED: {}", result.reason);
    } else {
        println!("✗ DENIED: {}", result.reason);
        if let Some(ref policy) = result.policy {
            println!("  Policy: {}", policy);
        }
    }
}

const PROMOTION_REQUESTER_ENV: &str = "AGILEPLUS_REQUESTED_BY";

fn resolve_promotion_requester<E>(
    requested_by: Option<String>,
    username: std::result::Result<String, E>,
) -> Result<String>
where
    E: std::error::Error + Send + Sync + 'static,
{
    if let Some(requested_by) = requested_by {
        return normalize_promotion_requester(requested_by, "promotion requester override");
    }

    let username = username.context(
        "unable to resolve promotion requester from OS username; pass --requested-by or set AGILEPLUS_REQUESTED_BY",
    )?;
    normalize_promotion_requester(username, "OS username")
}

fn normalize_promotion_requester(requested_by: String, source: &str) -> Result<String> {
    let requested_by = requested_by.trim();
    if requested_by.is_empty() {
        bail!("{source} must not be blank; pass --requested-by or set AGILEPLUS_REQUESTED_BY");
    }

    Ok(requested_by.to_string())
}

#[cfg(test)]
mod tests {
    use super::resolve_promotion_requester;

    #[test]
    fn resolve_promotion_requester_uses_explicit_override_when_lookup_fails() {
        assert_eq!(
            resolve_promotion_requester::<std::io::Error>(
                Some("release-approver".to_string()),
                Err(std::io::Error::other("user lookup unavailable")),
            )
            .unwrap(),
            "release-approver"
        );
    }

    #[test]
    fn resolve_promotion_requester_falls_back_to_os_username() {
        assert_eq!(
            resolve_promotion_requester::<std::io::Error>(None, Ok("release-agent".to_string()),)
                .unwrap(),
            "release-agent"
        );
    }

    #[test]
    fn resolve_promotion_requester_rejects_blank_override() {
        let error = resolve_promotion_requester::<std::io::Error>(
            Some("   ".to_string()),
            Ok("release-agent".to_string()),
        )
        .unwrap_err();

        assert!(error.to_string().contains("must not be blank"));
    }

    #[test]
    fn resolve_promotion_requester_guides_lookup_failure_without_override() {
        let error = resolve_promotion_requester::<std::io::Error>(
            None,
            Err(std::io::Error::other("user lookup unavailable")),
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("--requested-by or set AGILEPLUS_REQUESTED_BY"));
    }
}
