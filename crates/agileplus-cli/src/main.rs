//! AgilePlus CLI entry point — spec-driven development surface.
//!
//! Parses CLI arguments, initialises adapters, and routes to command handlers.
//! Platform health uses real HTTP/TCP probes via `agileplus-subcmds`.
//! Traceability: WP11-T060, T065 / WP12-T072 / WP14-T084..T087

mod agent_adapter;

use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use agent_adapter::RealAgentAdapter;
use agileplus_cli::commands::{
    cockpit::CockpitArgs, cycle::CycleArgs, dashboard::DashboardArgs, list::ListArgs,
    module::ModuleArgs, queue::QueueArgs, specify::SpecifyArgs,
};
#[cfg(feature = "full-deps")]
use agileplus_cli::commands::{
    implement::ImplementArgs, plan::PlanArgs, research::ResearchArgs,
    retrospective::RetrospectiveArgs, ship::ShipArgs, triage::TriageArgs, validate::ValidateArgs,
};
use agileplus_git::GitVcsAdapter;
use agileplus_sqlite::SqliteStorageAdapter;
use agileplus_subcmds::{PlatformArgs, run_platform};

/// Spec-driven development engine.
#[derive(Parser)]
#[command(name = "agileplus", version, about = "Spec-driven development engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Increase verbosity (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Path to SQLite database
    #[arg(long, global = true, default_value = ".agileplus/agileplus.db")]
    db: PathBuf,

    /// Path to git repository root (defaults to current directory)
    #[arg(long, global = true)]
    repo: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage cycles (time-boxed delivery units).
    Cycle(CycleArgs),
    /// List features in the database (optional filter by state).
    List(ListArgs),
    /// Create or revise a feature specification.
    Specify(SpecifyArgs),
    /// Manage the triage backlog queue.
    Queue(QueueArgs),
    /// Manage modules (product-area groupings of features).
    Module(ModuleArgs),
    /// Manage platform services (up, down, status, logs).
    Platform(PlatformArgs),
    /// Research a feature (codebase scan / feasibility).
    #[cfg(feature = "full-deps")]
    Research(ResearchArgs),
    /// Generate a delivery plan and work packages.
    #[cfg(feature = "full-deps")]
    Plan(PlanArgs),
    /// Implement work packages (dispatches agents).
    #[cfg(feature = "full-deps")]
    Implement(ImplementArgs),
    /// Validate governance evidence and policies.
    #[cfg(feature = "full-deps")]
    Validate(ValidateArgs),
    /// Ship a validated feature (merge + archive).
    #[cfg(feature = "full-deps")]
    Ship(ShipArgs),
    /// Generate a retrospective for a shipped feature.
    #[cfg(feature = "full-deps")]
    Retrospective(RetrospectiveArgs),
    /// Classify and route incoming items to the backlog.
    #[cfg(feature = "full-deps")]
    Triage(TriageArgs),
    /// Render an in-flight DAG / status dashboard from SQLite.
    Dashboard(DashboardArgs),
    /// Publish and read local cockpit scorecards.
    Cockpit(CockpitArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => tracing::Level::INFO,
        1 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .compact()
        .init();

    if let Err(e) = run(cli).await {
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}

fn open_vcs(repo: &Option<PathBuf>) -> Result<GitVcsAdapter> {
    match repo {
        Some(path) => Ok(GitVcsAdapter::new(path.clone())),
        None => GitVcsAdapter::from_current_dir()
            .context("Not inside a git repository. Run agileplus from your project root."),
    }
}

async fn run(cli: Cli) -> Result<()> {
    // Ensure DB directory exists for storage-backed commands
    if let Some(parent) = cli.db.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating directory {}", parent.display()))?;
        }
    }

    match cli.command {
        Commands::Platform(args) => run_platform(args),
        Commands::Cockpit(args) => agileplus_cli::commands::cockpit::run(&args),
        Commands::Dashboard(mut args) => {
            // Prefer global --db when the subcommand did not set its own path.
            if args.db.is_none() {
                args.db = Some(cli.db.clone());
            }
            agileplus_cli::commands::dashboard::run(&args)
        }
        Commands::Module(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            agileplus_cli::commands::module::run(args, &storage).await
        }
        Commands::Cycle(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            agileplus_cli::commands::cycle::run(args, &storage).await
        }
        Commands::Queue(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            agileplus_cli::commands::queue::run_queue(args, &storage).await
        }
        Commands::List(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            agileplus_cli::commands::list::run(args, &storage).await
        }
        Commands::Specify(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            agileplus_cli::commands::specify::run_specify(args, &storage, &vcs).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Research(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            agileplus_cli::commands::research::run_research(args, &storage, &vcs).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Plan(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            agileplus_cli::commands::plan::run_plan(args, &storage, &vcs).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Implement(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            let agent = RealAgentAdapter::new();
            agileplus_cli::commands::implement::run_implement(args, &storage, &vcs, &agent).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Validate(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            agileplus_cli::commands::validate::run_validate(args, &storage, &vcs).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Ship(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            agileplus_cli::commands::ship::run_ship(args, &storage, &vcs).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Retrospective(args) => {
            let storage = SqliteStorageAdapter::new(&cli.db)
                .with_context(|| format!("opening database at {}", cli.db.display()))?;
            let vcs = open_vcs(&cli.repo)?;
            agileplus_cli::commands::retrospective::run_retrospective(args, &storage, &vcs).await
        }
        #[cfg(feature = "full-deps")]
        Commands::Triage(args) => agileplus_cli::commands::triage::run_triage(args).await,
    }
}
