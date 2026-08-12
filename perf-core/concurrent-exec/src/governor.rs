//! Bounded admission control for agent and kernel work.
//!
//! The scheduler must not turn a burst of requests into an unbounded queue of
//! Tokio tasks.  `ResourceGovernor` accounts for both running and queued work,
//! validates request size before allocation, and applies a deadline while a
//! permit is waiting.  The permit releases its accounting on drop, including
//! cancellation paths.

use crate::ExecRequest;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, Copy)]
pub struct GovernorConfig {
    pub max_in_flight: usize,
    pub max_queue: usize,
    pub max_prompt_bytes: usize,
    pub max_tokens: usize,
    pub acquire_timeout: Duration,
}

impl GovernorConfig {
    pub fn for_concurrency(max_in_flight: usize) -> Self {
        let max_in_flight = max_in_flight.max(1);
        Self {
            max_in_flight,
            max_queue: max_in_flight.saturating_mul(4),
            max_prompt_bytes: 1 << 20,
            max_tokens: 32_768,
            acquire_timeout: Duration::from_secs(120),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GovernorError {
    #[error("resource queue is full (capacity={capacity})")]
    QueueFull { capacity: usize },
    #[error("request exceeds prompt byte budget ({actual} > {limit})")]
    PromptTooLarge { actual: usize, limit: usize },
    #[error("request exceeds token budget ({actual} > {limit})")]
    TokenBudgetExceeded { actual: usize, limit: usize },
    #[error("timed out waiting for a resource permit")]
    AcquireTimeout,
    #[error("resource governor is closed")]
    Closed,
}

/// A running or queued work permit. Dropping it always returns capacity.
pub struct GovernorPermit {
    _permit: OwnedSemaphorePermit,
    outstanding: Arc<AtomicUsize>,
}

impl Drop for GovernorPermit {
    fn drop(&mut self) {
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone)]
pub struct ResourceGovernor {
    config: GovernorConfig,
    permits: Arc<Semaphore>,
    outstanding: Arc<AtomicUsize>,
}

impl ResourceGovernor {
    pub fn new(config: GovernorConfig) -> Self {
        let max_in_flight = config.max_in_flight.max(1);
        Self {
            config: GovernorConfig {
                max_in_flight,
                ..config
            },
            permits: Arc::new(Semaphore::new(max_in_flight)),
            outstanding: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn config(&self) -> GovernorConfig {
        self.config
    }

    pub fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }

    fn validate(&self, request: &ExecRequest) -> Result<(), GovernorError> {
        let actual = request.prompt.len();
        if actual > self.config.max_prompt_bytes {
            return Err(GovernorError::PromptTooLarge {
                actual,
                limit: self.config.max_prompt_bytes,
            });
        }
        if request.max_tokens > self.config.max_tokens {
            return Err(GovernorError::TokenBudgetExceeded {
                actual: request.max_tokens,
                limit: self.config.max_tokens,
            });
        }
        Ok(())
    }

    fn reserve(&self) -> Result<(), GovernorError> {
        let capacity = self
            .config
            .max_in_flight
            .saturating_add(self.config.max_queue);
        let mut current = self.outstanding.load(Ordering::Acquire);
        loop {
            if current >= capacity {
                return Err(GovernorError::QueueFull { capacity });
            }
            match self.outstanding.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    pub async fn acquire(&self, request: &ExecRequest) -> Result<GovernorPermit, GovernorError> {
        self.validate(request)?;
        self.reserve()?;
        let result = tokio::time::timeout(
            self.config.acquire_timeout,
            self.permits.clone().acquire_owned(),
        )
        .await;
        let permit = match result {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                self.outstanding.fetch_sub(1, Ordering::AcqRel);
                return Err(GovernorError::Closed);
            }
            Err(_) => {
                self.outstanding.fetch_sub(1, Ordering::AcqRel);
                return Err(GovernorError::AcquireTimeout);
            }
        };
        Ok(GovernorPermit {
            _permit: permit,
            outstanding: self.outstanding.clone(),
        })
    }

    pub fn try_acquire(&self, request: &ExecRequest) -> Result<GovernorPermit, GovernorError> {
        self.validate(request)?;
        self.reserve()?;
        let permit = match self.permits.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.outstanding.fetch_sub(1, Ordering::AcqRel);
                return Err(GovernorError::QueueFull {
                    capacity: self.config.max_in_flight + self.config.max_queue,
                });
            }
        };
        Ok(GovernorPermit {
            _permit: permit,
            outstanding: self.outstanding.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ExecRequest {
        ExecRequest {
            prompt: "hello".into(),
            max_tokens: 8,
            temperature: 0.0,
            stop: vec![],
        }
    }

    #[tokio::test]
    async fn bounds_running_and_queued_work() {
        let governor = ResourceGovernor::new(GovernorConfig {
            max_in_flight: 1,
            max_queue: 1,
            ..GovernorConfig::for_concurrency(1)
        });
        let first = governor.acquire(&request()).await.unwrap();
        let pending = {
            let g = governor.clone();
            tokio::spawn(async move { g.acquire(&request()).await })
        };
        tokio::task::yield_now().await;
        assert_eq!(governor.outstanding(), 2);
        assert!(matches!(
            governor.try_acquire(&request()),
            Err(GovernorError::QueueFull { .. })
        ));
        drop(first);
        let second = pending.await.unwrap().unwrap();
        assert_eq!(governor.outstanding(), 1);
        drop(second);
        assert_eq!(governor.outstanding(), 0);
    }

    #[tokio::test]
    async fn rejects_oversized_requests_before_reserving() {
        let governor = ResourceGovernor::new(GovernorConfig {
            max_prompt_bytes: 2,
            max_tokens: 4,
            ..GovernorConfig::for_concurrency(1)
        });
        let mut req = request();
        assert!(matches!(
            governor.acquire(&req).await,
            Err(GovernorError::PromptTooLarge { .. })
        ));
        req.prompt = "ok".into();
        req.max_tokens = 5;
        assert!(matches!(
            governor.acquire(&req).await,
            Err(GovernorError::TokenBudgetExceeded { .. })
        ));
        assert_eq!(governor.outstanding(), 0);
    }

    #[tokio::test]
    async fn timeout_releases_outstanding_reservation() {
        let governor = ResourceGovernor::new(GovernorConfig {
            max_in_flight: 1,
            max_queue: 1,
            acquire_timeout: Duration::from_millis(1),
            ..GovernorConfig::for_concurrency(1)
        });
        let first = governor.acquire(&request()).await.unwrap();
        assert!(matches!(
            governor.acquire(&request()).await,
            Err(GovernorError::AcquireTimeout)
        ));
        assert_eq!(governor.outstanding(), 1);
        drop(first);
        assert_eq!(governor.outstanding(), 0);
    }
}
