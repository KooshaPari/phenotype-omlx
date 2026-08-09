# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `apps/bench-cockpit/scripts/evals/setup_langfuse_cloud.py` and `setup_langfuse_judges.py` now use a hardened `_load_dotenv()` that strips surrounding quotes and skips comments / blank lines, so values like `LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"` are picked up verbatim rather than keeping the literal quotes that broke LANGFUSE_HOST resolution on Aug 8.
