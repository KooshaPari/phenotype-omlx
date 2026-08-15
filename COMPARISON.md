# Comparison Matrix

## Feature Comparison

This document compares **phenotype-shared** with similar tools in the Rust infrastructure toolkit space.

**Relationship to [`phenotype-infrakit`](../phenotype-infrakit)**: this workspace **includes** the same four infra crates (event sourcing, cache adapter, policy engine, state machine) and **adds** domain, application, port interfaces, and HTTP/PostgreSQL/Redis adapters. For a slimmer dependency set, depend on **infrakit** only; use **shared** when you need the full hex stack in one repo.
