use crate::{TaskSpec, Suite};
/// PPL scoring — calibration-free quality metric.
pub fn score_perplexity(log_probs: &[f64]) -> f64 {
    if log_probs.is_empty() { return f64::INFINITY; }
    let nll: f64 = log_probs.iter().sum();
    (-nll / log_probs.len() as f64).exp()
}

pub fn load_tasks() -> Vec<TaskSpec> {
    vec![
        TaskSpec {
            id: "ppl_sentence_1".into(), suite: Suite::Perplexity,
            prompt: "The quick brown fox jumps over the lazy dog.".into(),
            expected: None, choices: None,
        },
    ]
}
