//! No‑op adapter for [`TraceabilityPort`].
//!
//! Returns `Ok(())` / `Ok(vec![])` for every call. Useful when no external traceability
//! system is connected, in integration tests, or as a default/fallback implementation.

use async_trait::async_trait;

use crate::error::DomainError;
use crate::ports::traceability_port::TraceabilityPort;
use crate::traceability::TraceRef;

/// A [`TraceabilityPort`] that accepts every link and always returns an empty trace list.
///
/// # Examples
///
/// ```rust,ignore
/// // Doctest ignored: the example body is pre-migration (uses TraceRef.entity_id,
/// // which no longer exists). See the unit tests in tests/traceability_test.rs
/// // (currently `#[ignore]`d as a whole file) for the current API contract.
/// use agileplus_domain::adapters::noop_trace_adapter::NoopTraceAdapter;
/// use agileplus_domain::ports::traceability_port::TraceabilityPort;
/// use agileplus_domain::traceability::TraceRef;
/// use chrono::Utc;
///
/// # tokio_test::block_on(async {
/// let adapter = NoopTraceAdapter;
/// let entity_id = format!("trace-{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));
/// let trace_ref = TraceRef {
///     trace_id: "FR-001".into(),
///     artifact_type: "requirement".into(),
///     linked_at: Utc::now(),
/// };
///
/// let link = adapter.link_trace(entity_id, trace_ref).await;
/// assert!(link.is_ok());
///
/// let traces = adapter.get_traces(entity_id).await;
/// assert!(traces.is_ok());
/// assert!(traces.unwrap().is_empty());
/// # })
/// ```
pub struct NoopTraceAdapter;

#[async_trait]
impl TraceabilityPort for NoopTraceAdapter {
    async fn link_trace(
        &self,
        _entity_id: String,
        _trace_ref: TraceRef,
    ) -> Result<(), DomainError> {
        Ok(())
    }

    async fn get_traces(&self, _entity_id: String) -> Result<Vec<TraceRef>, DomainError> {
        Ok(vec![])
    }
}
