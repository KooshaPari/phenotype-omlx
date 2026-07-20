use crate::{TaskSpec, Suite};
pub fn load_tasks() -> Vec<TaskSpec> {
    vec![
        TaskSpec {
            id: "mmlu_anatomy_1".into(), suite: Suite::MMLU,
            prompt: "The heart is located in which cavity of the human body?\nA) Cranial\nB) Thoracic\nC) Abdominal\nD) Pelvic\nAnswer:".into(),
            expected: Some("B".into()), choices: None,
        },
        TaskSpec {
            id: "mmlu_physics_1".into(), suite: Suite::MMLU,
            prompt: "What is the SI unit of force?\nA) Joule\nB) Newton\nC) Watt\nD) Pascal\nAnswer:".into(),
            expected: Some("B".into()), choices: None,
        },
    ]
}
