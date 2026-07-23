use crate::candidate::Candidate;
use crate::quality::{evaluate_for_production, QualityAttachment, QualityError};
use crate::record::TuningRecord;
use crate::selector::RejectionReason;

/// Verify that `record`'s quality attachment (if any) lets `candidate`
/// serve under `SelectionPolicy::Production`.
///
/// Translation rules:
/// - No attachment -> `RejectionReason::MissingQualityEvidence` describing
///   what needs to be attached.
/// - Attachment present but empty/with `gates.is_empty()` -> same rejection.
/// - Attachment present but a gate failed -> `QualityGateFailed`.
/// - Signature/duplicate errors are reported as `Other` (the registry
///   surface doesn't yet know how to react to them).
pub(crate) fn check_production_quality(
    candidate: &Candidate,
    record: &TuningRecord,
) -> std::result::Result<(), RejectionReason> {
    let attachment: &QualityAttachment = match record.quality.as_ref() {
        Some(q) => q,
        None => {
            return Err(RejectionReason::MissingQualityEvidence(format!(
                "candidate {} has no quality attachment under Production policy",
                candidate.id
            )))
        }
    };
    match evaluate_for_production(record, attachment) {
        Ok(()) => Ok(()),
        Err(QualityError::PromotionGateMissingEvidence { gate }) => {
            Err(RejectionReason::MissingQualityEvidence(format!(
                "candidate {} missing evidence for gate '{}'",
                candidate.id, gate
            )))
        }
        Err(QualityError::PromotionGateRejected {
            gate,
            observed,
            threshold,
        }) => Err(RejectionReason::QualityGateFailed {
            gate,
            observed,
            threshold,
        }),
        Err(QualityError::PromotionWithoutGates) => {
            Err(RejectionReason::MissingQualityEvidence(format!(
                "candidate {} attachment has no gates configured",
                candidate.id
            )))
        }
        Err(e) => Err(RejectionReason::Other(format!(
            "candidate {} quality check failed: {}",
            candidate.id, e
        ))),
    }
}
