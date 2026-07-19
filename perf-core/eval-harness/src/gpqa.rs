use crate::{TaskSpec, Suite};
pub fn load_tasks() -> Vec<TaskSpec> {
    vec![
        TaskSpec {
            id: "gpqa_chem_1".into(), suite: Suite::GPQA,
            prompt: "Which of the following has the highest electronegativity?\nA) Fluorine\nB) Oxygen\nC) Nitrogen\nD) Chlorine\nAnswer:".into(),
            expected: Some("A".into()), choices: None,
        },
        TaskSpec {
            id: "gpqa_bio_1".into(), suite: Suite::GPQA,
            prompt: "Which organelle is responsible for protein synthesis in eukaryotic cells?\nA) Mitochondria\nB) Ribosome\nC) Golgi apparatus\nD) Lysosome\nAnswer:".into(),
            expected: Some("B".into()), choices: None,
        },
    ]
}
