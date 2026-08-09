//! User aggregate — a person who interacts with the system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

use crate::error::DomainError;

/// Role of a user within the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    Member,
    Viewer,
}

impl fmt::Display for UserRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UserRole::Admin => "admin",
            UserRole::Member => "member",
            UserRole::Viewer => "viewer",
        };
        write!(f, "{s}")
    }
}

impl FromStr for UserRole {
    type Err = DomainError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "admin" => Ok(UserRole::Admin),
            "member" => Ok(UserRole::Member),
            "viewer" => Ok(UserRole::Viewer),
            _ => Err(DomainError::Validation(format!("unknown UserRole: {s}"))),
        }
    }
}

/// Account status of a user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserStatus {
    Active,
    Inactive,
    Suspended,
}

impl fmt::Display for UserStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            UserStatus::Active => "active",
            UserStatus::Inactive => "inactive",
            UserStatus::Suspended => "suspended",
        };
        write!(f, "{s}")
    }
}

impl FromStr for UserStatus {
    type Err = DomainError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "active" => Ok(UserStatus::Active),
            "inactive" => Ok(UserStatus::Inactive),
            "suspended" => Ok(UserStatus::Suspended),
            _ => Err(DomainError::Validation(format!("unknown UserStatus: {s}"))),
        }
    }
}

impl UserStatus {
    /// Returns `true` if a transition from `self` to `target` is allowed.
    pub fn can_transition_to(self, target: UserStatus) -> bool {
        use UserStatus::*;
        matches!(
            (self, target),
            (Active, Inactive) | (Active, Suspended) | (Inactive, Active) | (Suspended, Active)
        )
    }
}

/// A user who interacts with the platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    /// Display name — must be non-empty.
    pub display_name: String,
    /// Email address — must contain '@'.
    pub email: String,
    pub role: UserRole,
    pub status: UserStatus,
    pub avatar_url: Option<String>,
    pub github_login: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl User {
    /// Construct a new `User`, enforcing non-empty display name and basic email
    /// validity (must contain `@`).
    pub fn new(display_name: &str, email: &str, role: UserRole) -> Result<Self, DomainError> {
        let display_name = display_name.trim();
        if display_name.is_empty() {
            return Err(DomainError::Validation(
                "display_name must not be empty".to_string(),
            ));
        }
        let email = email.trim();
        if !email.contains('@') {
            return Err(DomainError::Validation(
                "email must contain '@'".to_string(),
            ));
        }
        let now = Utc::now();
        Ok(Self {
            id: 0,
            display_name: display_name.to_string(),
            email: email.to_string(),
            role,
            status: UserStatus::Active,
            avatar_url: None,
            github_login: None,
            created_at: now,
            updated_at: now,
        })
    }

    /// Attempt a status transition. Returns `Err(DomainError::InvalidTransition)`
    /// if the transition is not permitted.
    pub fn transition_status(&mut self, target: UserStatus) -> Result<(), DomainError> {
        if !self.status.can_transition_to(target) {
            return Err(DomainError::InvalidTransition {
                from: self.status.to_string(),
                to: target.to_string(),
                reason: "not an allowed user status transition".to_string(),
            });
        }
        self.status = target;
        self.updated_at = Utc::now();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_user_construction() {
        let u = User::new("Alice", "alice@example.com", UserRole::Member).unwrap();
        assert_eq!(u.display_name, "Alice");
        assert_eq!(u.email, "alice@example.com");
        assert_eq!(u.status, UserStatus::Active);
        assert_eq!(u.role, UserRole::Member);
    }

    #[test]
    fn rejects_empty_display_name() {
        let err = User::new("  ", "a@b.com", UserRole::Viewer).unwrap_err();
        assert!(matches!(err, DomainError::Validation(_)));
    }

    #[test]
    fn rejects_invalid_email() {
        let err = User::new("Bob", "notanemail", UserRole::Member).unwrap_err();
        assert!(matches!(err, DomainError::Validation(_)));
    }

    #[test]
    fn valid_status_transition() {
        let mut u = User::new("Carol", "carol@x.com", UserRole::Admin).unwrap();
        u.transition_status(UserStatus::Inactive).unwrap();
        assert_eq!(u.status, UserStatus::Inactive);
    }

    #[test]
    fn invalid_status_transition_rejected() {
        let mut u = User::new("Dave", "dave@x.com", UserRole::Member).unwrap();
        // Active -> Suspended is allowed; Suspended -> Inactive is NOT
        u.transition_status(UserStatus::Suspended).unwrap();
        let err = u.transition_status(UserStatus::Inactive).unwrap_err();
        assert!(matches!(err, DomainError::InvalidTransition { .. }));
    }
}
