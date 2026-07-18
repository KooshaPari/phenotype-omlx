use crate::{TaskSpec, Suite};
pub fn load_tasks() -> Vec<TaskSpec> {
    vec![
        TaskSpec {
            id: "tb_cd_1".into(), suite: Suite::TerminalBench,
            prompt: "Change to the directory `/home/user/projects` and list all files ending in '.py'.\n\nCommands:".into(),
            expected: Some("cd /home/user/projects && ls *.py".into()), choices: None,
        },
        TaskSpec {
            id: "tb_grep_1".into(), suite: Suite::TerminalBench,
            prompt: "Find all files containing the string 'FIXME' in the current directory and its subdirectories.\n\nCommands:".into(),
            expected: Some("grep -r 'FIXME'".into()), choices: None,
        },
    ]
}
