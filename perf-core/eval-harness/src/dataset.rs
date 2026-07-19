//! Dataset: a suite-attributed, provenance-tagged collection of [`TaskSpec`]s.
//!
//! Every loader returns a [`Dataset`] rather than a bare `Vec<TaskSpec>` so
//! that the upstream source, revision, split, and content hash travel with the
//! tasks they produced. Reports and downstream consumers can then attribute
//! results to the exact dataset bytes that were evaluated.
//!
//! `Dataset` implements [`Deref<Target = [TaskSpec]>`] so existing call sites
//! that iterate, slice, or index the underlying tasks continue to work.

use crate::provenance::DatasetProvenance;
use crate::{Suite, TaskSpec};
use serde::{Deserialize, Serialize};
use std::ops::Deref;

/// A loaded evaluation dataset: tasks together with their provenance.
///
/// The struct is intentionally not generic over the loader. Downstream
/// consumers can deserialize a dataset from JSON (e.g. for archival) by
/// round-tripping [`serde_json`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dataset {
    /// Suite this dataset belongs to. Must agree with every contained task.
    pub suite: Suite,
    /// Explicit provenance for the dataset bytes that produced these tasks.
    pub provenance: DatasetProvenance,
    /// Loaded tasks in stable order (loaders sort by task id).
    pub tasks: Vec<TaskSpec>,
}

impl Dataset {
    /// Construct a new dataset. `provenance.task_count` is overwritten with
    /// `tasks.len()` so the recorded count always matches the actual size.
    pub fn new(suite: Suite, mut provenance: DatasetProvenance, tasks: Vec<TaskSpec>) -> Self {
        provenance.task_count = tasks.len();
        Self {
            suite,
            provenance,
            tasks,
        }
    }

    /// Suite the dataset belongs to.
    pub fn suite(&self) -> Suite {
        self.suite
    }

    /// Explicit dataset provenance.
    pub fn provenance(&self) -> &DatasetProvenance {
        &self.provenance
    }

    /// Consume the dataset and return the underlying tasks, discarding
    /// provenance. Useful when callers have already recorded provenance and
    /// only need the task list.
    pub fn into_tasks(self) -> Vec<TaskSpec> {
        self.tasks
    }

    /// Borrow the underlying tasks.
    pub fn as_tasks(&self) -> &[TaskSpec] {
        &self.tasks
    }
}

impl Deref for Dataset {
    type Target = [TaskSpec];

    fn deref(&self) -> &Self::Target {
        &self.tasks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DatasetProvenance;

    fn sample_provenance(task_count: usize) -> DatasetProvenance {
        DatasetProvenance::new("test-source", "v1", "test", b"abc", task_count)
    }

    fn sample_task(id: &str, suite: Suite) -> TaskSpec {
        TaskSpec {
            id: id.into(),
            suite,
            prompt: format!("prompt-{id}"),
            expected: Some("A".into()),
            choices: vec!["a".into(), "b".into()],
            criteria: None,
        }
    }

    #[test]
    fn deref_allows_iteration_and_indexing() {
        let dataset = Dataset::new(
            Suite::MMLU,
            sample_provenance(2),
            vec![sample_task("b", Suite::MMLU), sample_task("a", Suite::MMLU)],
        );
        // Iteration through Deref.
        let ids: Vec<&str> = dataset.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, vec!["b", "a"]);
        // Indexing through Deref.
        assert_eq!(dataset[0].id, "b");
    }

    #[test]
    fn new_overwrites_task_count_with_actual_size() {
        // Caller passes a stale task_count; Dataset::new corrects it so the
        // recorded provenance matches the actual task list.
        let dataset = Dataset::new(
            Suite::MMLU,
            sample_provenance(99),
            vec![sample_task("a", Suite::MMLU)],
        );
        assert_eq!(dataset.provenance.task_count, 1);
    }

    #[test]
    fn serde_round_trips() {
        let dataset = Dataset::new(
            Suite::GPQA,
            sample_provenance(1),
            vec![sample_task("gpqa_a", Suite::GPQA)],
        );
        let encoded = serde_json::to_string(&dataset).unwrap();
        let decoded: Dataset = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, dataset);
    }

    #[test]
    fn into_tasks_drops_provenance() {
        let dataset = Dataset::new(
            Suite::MMLU,
            sample_provenance(1),
            vec![sample_task("a", Suite::MMLU)],
        );
        let tasks = dataset.into_tasks();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "a");
    }
}
