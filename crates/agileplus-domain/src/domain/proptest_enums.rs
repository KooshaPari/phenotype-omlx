//! Property-based tests for domain enum FromStr implementations.
//!
//! Verifies that:
//! 1. Anything `FromStr` accepts round-trips through serde
//! 2. Strings with special characters / digits are always rejected
//! 3. The parsers never panic on arbitrary input

use std::str::FromStr;

use crate::domain::backlog::{BacklogPriority, BacklogStatus, Intent};
use crate::domain::epic::EpicStatus;
use crate::domain::user::{UserRole, UserStatus};

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn user_role_round_trip(s in "[a-zA-Z_-]{0,32}") {
            // Whatever FromStr accepts must round-trip via serialize+deserialize
            if let Ok(role) = UserRole::from_str(&s) {
                let json = serde_json::to_string(&role).unwrap();
                let back: UserRole = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(role, back);
            }
        }

        #[test]
        fn user_status_round_trip(s in "[a-zA-Z_]{0,32}") {
            if let Ok(status) = UserStatus::from_str(&s) {
                let json = serde_json::to_string(&status).unwrap();
                let back: UserStatus = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(status, back);
            }
        }

        #[test]
        fn intent_round_trip(s in "[a-zA-Z_-]{0,32}") {
            if let Ok(intent) = Intent::from_str(&s) {
                let json = serde_json::to_string(&intent).unwrap();
                let back: Intent = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(intent, back);
            }
        }

        #[test]
        fn backlog_priority_round_trip(s in "[a-zA-Z_]{0,32}") {
            if let Ok(p) = BacklogPriority::from_str(&s) {
                let json = serde_json::to_string(&p).unwrap();
                let back: BacklogPriority = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(p, back);
            }
        }

        #[test]
        fn backlog_status_round_trip(s in "[a-zA-Z_]{0,32}") {
            if let Ok(s2) = BacklogStatus::from_str(&s) {
                let json = serde_json::to_string(&s2).unwrap();
                let back: BacklogStatus = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(s2, back);
            }
        }

        #[test]
        fn epic_status_round_trip(s in "[a-zA-Z_-]{0,32}") {
            if let Ok(es) = EpicStatus::from_str(&s) {
                let json = serde_json::to_string(&es).unwrap();
                let back: EpicStatus = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(es, back);
            }
        }

        #[test]
        fn unknown_strings_rejected(s in "[!@#$%^&*()0-9 ]{1,32}") {
            // Special chars and digits should never be accepted as enum values
            prop_assert!(UserRole::from_str(&s).is_err());
            prop_assert!(UserStatus::from_str(&s).is_err());
            prop_assert!(Intent::from_str(&s).is_err());
            prop_assert!(BacklogPriority::from_str(&s).is_err());
            prop_assert!(BacklogStatus::from_str(&s).is_err());
            prop_assert!(EpicStatus::from_str(&s).is_err());
        }
    }
}
