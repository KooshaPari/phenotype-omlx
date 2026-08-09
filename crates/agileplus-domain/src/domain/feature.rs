// SPDX-License-Identifier: MIT OR Apache-2.0
//! Feature aggregate — the central planning unit.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::state_machine::FeatureState;

pub(crate) mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        serializer.serialize_str(&out)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.len() != 64 {
            return Err(serde::de::Error::invalid_length(
                value.len(),
                &"64 hex chars",
            ));
        }

        let mut bytes = [0_u8; 32];
        for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = hex_value(chunk[0]).map_err(serde::de::Error::custom)?;
            let low = hex_value(chunk[1]).map_err(serde::de::Error::custom)?;
            bytes[index] = (high << 4) | low;
        }
        Ok(bytes)
    }

    fn hex_value(byte: u8) -> Result<u8, &'static str> {
        match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            b'A'..=b'F' => Ok(byte - b'A' + 10),
            _ => Err("invalid hex digit"),
        }
    }
}

/// A software feature tracked through the planning lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    pub id: i64,
    pub slug: String,
    pub friendly_name: String,
    pub state: FeatureState,
    pub spec_hash: [u8; 32],
    pub target_branch: String,
    pub plane_issue_id: Option<String>,
    pub plane_state_id: Option<String>,
    pub labels: Vec<String>,
    pub module_id: Option<i64>,
    pub project_id: Option<i64>,
    pub created_at_commit: Option<String>,
    pub last_modified_commit: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Feature {
    /// Derive a kebab-case slug from a display name.
    pub fn slug_from_name(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Attempt a state transition. Returns an error string if the transition is not allowed.
    pub fn transition(&mut self, target: FeatureState) -> Result<(), String> {
        use FeatureState::*;
        let allowed = match self.state {
            Created => matches!(target, Specified),
            Specified => matches!(target, Researched),
            Researched => matches!(target, Planned),
            Planned => matches!(target, Implementing),
            Implementing => matches!(target, Validated),
            Validated => matches!(target, Shipped),
            Shipped => matches!(target, Retrospected),
            Retrospected => false,
        };
        if allowed {
            self.state = target;
            self.updated_at = Utc::now();
            Ok(())
        } else {
            Err(format!(
                "invalid transition {:?} -> {:?}",
                self.state, target
            ))
        }
    }

    /// Construct a new Feature with sensible defaults.
    pub fn new(
        slug: &str,
        friendly_name: &str,
        spec_hash: [u8; 32],
        target_branch: Option<&str>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: 0,
            slug: slug.to_string(),
            friendly_name: friendly_name.to_string(),
            state: FeatureState::Created,
            spec_hash,
            target_branch: target_branch.unwrap_or("main").to_string(),
            plane_issue_id: None,
            plane_state_id: None,
            labels: Vec::new(),
            module_id: None,
            project_id: None,
            created_at_commit: None,
            last_modified_commit: None,
            created_at: now,
            updated_at: now,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_transition_updates_state() {
        let mut feature = Feature::new("auth", "Authentication", [0; 32], None);

        feature
            .transition(FeatureState::Specified)
            .expect("domain operation");

        assert_eq!(feature.state, FeatureState::Specified);
    }

    #[test]
    fn invalid_transition_is_rejected_without_mutating_state() {
        let mut feature = Feature::new("auth", "Authentication", [0; 32], None);

        let err = feature.transition(FeatureState::Shipped).unwrap_err();

        assert_eq!(feature.state, FeatureState::Created);
        assert!(err.contains("invalid transition Created -> Shipped"));
    }
}
