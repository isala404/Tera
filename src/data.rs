//! Every prompt, instruction file and config template tera ships, embedded from
//! `data/` at build time.
//!
//! `data/` is the single source: nothing here is authored in Rust. Prompts used
//! to live inside `format!` calls scattered across the modules that sent them,
//! which meant editing a prompt was a code change, the shell and jq examples had
//! to have every brace doubled to survive `format!`, and there was nowhere to
//! read the assistant's instructions as a whole. Now they are ordinary files;
//! `include_str!` bakes them into the binary so a deployed tera still needs
//! nothing but itself, and Cargo rebuilds when one of them changes.
//!
//! Substitution is `{{PLACEHOLDER}}` and plain string replacement. See
//! [`render`]. Deliberately not a template engine: paths are the only thing
//! interpolated, and single braces stay literal so jq and JSON examples can be
//! written the way they are typed.

/// Machine-owned instruction files start with this. Detection is by prefix, not
/// by the whole line: the marker names the product and says where user edits
/// belong, so its wording changes, and a wording change must not make the
/// daemon mistake its own file for the user's and move it aside.
pub const GENERATED_MARKER_PREFIX: &str = "<!-- generated:";

// Instruction files installed into the workspace.
pub const WORKSPACE_AGENTS: &str = include_str!("../data/workspace/AGENTS.md");
pub const PERSONA: &str = include_str!("../data/workspace/PERSONA.md");
/// Seeded once, then owned by the agent. Not a generated file.
pub const SYSTEM_NOTES: &str = include_str!("../data/workspace/SYSTEM.md");
/// Craft: how to work, what to reach for, model tiers, the ways it has gone wrong
/// before. Split out of `AGENTS.md` so a session that only answers a question does
/// not pay for it.
pub const WORKING: &str = include_str!("../data/workspace/WORKING.md");
pub const CODEX_HOME_AGENTS: &str = include_str!("../data/workspace/codex-home/AGENTS.md");
pub const PROJECTS_AGENTS: &str = include_str!("../data/workspace/projects/AGENTS.md");
pub const TASKS_AGENTS: &str = include_str!("../data/workspace/tasks/AGENTS.md");
pub const SCHEDULE_AGENTS: &str = include_str!("../data/workspace/tasks/SCHEDULE_AGENTS.md");
pub const HISTORY_SCHEMA: &str = include_str!("../data/workspace/history/SCHEMA.md");
pub const LOGS_SCHEMA: &str = include_str!("../data/workspace/logs/SCHEMA.md");

/// A built-in skill package. Skill files are embedded so a released tera can
/// initialize a workspace without depending on the source tree being present.
pub struct BuiltinSkill {
    pub name: &'static str,
    pub files: &'static [BuiltinSkillFile],
}

pub struct BuiltinSkillFile {
    pub relative_path: &'static str,
    pub contents: &'static [u8],
    pub executable: bool,
}

// The build script discovers every direct child of data/skills and generates
// this manifest. Adding a built-in skill is therefore a data change, not a Rust
// change.
include!(concat!(env!("OUT_DIR"), "/builtin_skills.rs"));

// Prompts sent to a model.
pub const MEMORY_OPTIMIZER_PROMPT: &str = include_str!("../data/prompts/memory-optimizer.md");
pub const MEMORY_REBUILD_PROMPT: &str = include_str!("../data/prompts/memory-rebuild.md");
pub const SCHEDULED_TASK_PROMPT: &str = include_str!("../data/prompts/scheduled-task.md");
pub const SCHEDULED_TASK_LATE_NOTE: &str = include_str!("../data/prompts/scheduled-task-late.md");
/// The seeded machine-health schedule. Its own prompt rather than something the
/// agent has to compose, so a fresh workspace looks after the host from day one.
pub const SELF_CARE_PROMPT: &str = include_str!("../data/prompts/self-care.md");

// Config we generate for other processes.
pub const CODEX_CONFIG_TOML: &str = include_str!("../data/config/codex-config.toml");
pub const MCP_TOOLS_JSON: &str = include_str!("../data/config/mcp-tools.json");

/// Substitute `{{NAME}}` placeholders.
///
/// A placeholder with no value stays in the output rather than becoming an empty
/// string, a prompt that still says `{{STAGING}}` is obviously broken, whereas
/// one that tells a model to work in `` is subtly broken. The test below is what
/// keeps that from reaching a running daemon.
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (name, value) in vars {
        out = out.replace(&format!("{{{{{name}}}}}"), value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: &[(&str, &str)] = &[
        ("README.md", include_str!("../README.md")),
        ("workspace/AGENTS.md", WORKSPACE_AGENTS),
        ("workspace/PERSONA.md", PERSONA),
        ("workspace/SYSTEM.md", SYSTEM_NOTES),
        ("workspace/WORKING.md", WORKING),
        ("workspace/codex-home/AGENTS.md", CODEX_HOME_AGENTS),
        ("workspace/projects/AGENTS.md", PROJECTS_AGENTS),
        ("workspace/tasks/AGENTS.md", TASKS_AGENTS),
        ("workspace/tasks/SCHEDULE_AGENTS.md", SCHEDULE_AGENTS),
        ("workspace/history/SCHEMA.md", HISTORY_SCHEMA),
        ("workspace/logs/SCHEMA.md", LOGS_SCHEMA),
        ("prompts/memory-optimizer.md", MEMORY_OPTIMIZER_PROMPT),
        ("prompts/memory-rebuild.md", MEMORY_REBUILD_PROMPT),
        ("prompts/scheduled-task.md", SCHEDULED_TASK_PROMPT),
        ("prompts/scheduled-task-late.md", SCHEDULED_TASK_LATE_NOTE),
        ("prompts/self-care.md", SELF_CARE_PROMPT),
        ("skills/spotify/SKILL.md", include_str!("../data/skills/spotify/SKILL.md")),
        ("config/codex-config.toml", CODEX_CONFIG_TOML),
        ("config/mcp-tools.json", MCP_TOOLS_JSON),
    ];

    /// Every placeholder some call site actually fills in.
    const SUPPLIED: &[&str] = &[
        "WORKSPACE", "STAGING", "MEMORIES", "HISTORY", "JSONL", "SQLITE", "ASSETS", "SCHEMA",
        "EVENTS", "TASK_NAME", "SCHEDULE_ID", "NOW", "TASK_DIR", "LATE_NOTE", "TASK_PROMPT",
        "LATE_MINUTES", "MISSED", "BIN", "SOCKET", "MODEL", "EFFORT", "OWNER",
    ];

    fn placeholders_in(text: &str) -> Vec<String> {
        let mut found = Vec::new();
        let mut rest = text;
        while let Some(start) = rest.find("{{") {
            let after = &rest[start + 2..];
            match after.find("}}") {
                Some(end) => {
                    found.push(after[..end].to_string());
                    rest = &after[end + 2..];
                }
                None => break,
            }
        }
        found
    }

    fn markdown_prose(text: &str) -> Vec<(usize, String)> {
        let mut prose = Vec::new();
        let mut in_fence = false;
        let mut in_frontmatter = false;

        for (index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if index == 0 && trimmed == "---" {
                in_frontmatter = true;
                continue;
            }
            if in_frontmatter {
                if trimmed == "---" {
                    in_frontmatter = false;
                }
                continue;
            }
            if line.trim_start().starts_with("```") {
                in_fence = !in_fence;
                continue;
            }
            if in_fence || trimmed.starts_with("<!--") {
                continue;
            }
            if !trimmed.is_empty()
                && trimmed
                    .chars()
                    .all(|character| matches!(character, '|' | ' ' | ':' | '-'))
            {
                continue;
            }

            let mut plain = String::new();
            let mut in_inline_code = false;
            let mut in_link_target = false;
            let mut previous = None;
            for character in line.chars() {
                if character == '`' {
                    in_inline_code = !in_inline_code;
                } else if !in_inline_code && character == '(' && previous == Some(']') {
                    in_link_target = true;
                } else if in_link_target {
                    if character == ')' {
                        in_link_target = false;
                    }
                } else if !in_inline_code {
                    plain.push(character);
                }
                previous = Some(character);
            }
            let plain = plain.trim_start();
            let plain = plain.strip_prefix("- ").unwrap_or(plain);
            if !plain.is_empty() {
                prose.push((index + 1, plain.to_string()));
            }
        }

        prose
    }

    /// The files that address the user by name must keep doing it through
    /// `{{OWNER}}`.
    ///
    /// This is a general tool: a name written into `data/` would make every other
    /// install's assistant address its user as somebody else. Writing "Ada asked
    /// for X" reads more naturally than writing the placeholder, so the mistake is
    /// easy to make and invisible once made.
    ///
    /// A whitelist rather than a heuristic, because "their" legitimately refers to
    /// parallel workers in `projects/AGENTS.md` and to nobody in particular
    /// elsewhere. This catches a placeholder being *removed*; the pronoun test
    /// below is what catches a name being *added*, since a literal name almost
    /// always arrives with a gendered pronoun beside it.
    #[test]
    fn test_the_files_that_address_the_owner_still_do_it_by_placeholder() {
        const MUST_NAME_THE_OWNER: &[&str] = &[
            "workspace/AGENTS.md",
            "workspace/PERSONA.md",
            "workspace/SYSTEM.md",
            "workspace/tasks/AGENTS.md",
            "workspace/tasks/SCHEDULE_AGENTS.md",
            "prompts/memory-optimizer.md",
            "prompts/memory-rebuild.md",
            "prompts/scheduled-task.md",
            "prompts/self-care.md",
            "config/mcp-tools.json",
        ];

        for wanted in MUST_NAME_THE_OWNER {
            let (_, text) = ALL
                .iter()
                .find(|(name, _)| name == wanted)
                .unwrap_or_else(|| panic!("{wanted} is not in ALL"));
            assert!(
                text.contains("{{OWNER}}"),
                "{wanted} no longer addresses the owner through {{{{OWNER}}}}"
            );
        }
    }

    /// Gendered pronouns are a hardcoded assumption about a stranger, exactly like
    /// a hardcoded name. `AGENTS.md` in particular used to be written throughout in
    /// he/him.
    #[test]
    fn test_shipped_text_does_not_assume_the_owners_gender() {
        const GENDERED: &[&str] = &[
            " he ", " He ", " him ", " him.", " his ", " His ", " she ", " She ", " her ", " hers ",
        ];

        for (name, text) in ALL {
            for word in GENDERED {
                assert!(
                    !text.contains(word),
                    "{name} contains {word:?}; the owner's pronouns are not ours to assume"
                );
            }
        }
    }

    /// A typo in a placeholder name is invisible until a model is staring at
    /// `{{WORKSPCE}}`, so it is caught here instead.
    #[test]
    fn test_every_placeholder_is_one_a_call_site_supplies() {
        for (name, text) in ALL {
            for placeholder in placeholders_in(text) {
                assert!(
                    SUPPLIED.contains(&placeholder.as_str()),
                    "{name} uses {{{{{placeholder}}}}}, which nothing supplies"
                );
            }
        }
    }

    #[test]
    fn test_render_substitutes_and_leaves_single_braces_alone() {
        let rendered = render("cd {{WORKSPACE}} && jq '{t, from}'", &[("WORKSPACE", "/ws")]);
        assert_eq!(rendered, "cd /ws && jq '{t, from}'");
    }

    /// The generated files must be recognisable as ours, or workspace init will
    /// treat them as hand-written and shuffle them aside on every start.
    #[test]
    fn test_generated_instruction_files_carry_the_marker() {
        for text in [
            WORKSPACE_AGENTS,
            CODEX_HOME_AGENTS,
            PROJECTS_AGENTS,
            TASKS_AGENTS,
            SCHEDULE_AGENTS,
            HISTORY_SCHEMA,
            LOGS_SCHEMA,
            WORKING,
        ] {
            assert!(text.starts_with(GENERATED_MARKER_PREFIX), "{text:.60}");
        }
        // These two are not ours to rewrite, and must never look generated.
        assert!(!PERSONA.starts_with(GENERATED_MARKER_PREFIX));
        assert!(!SYSTEM_NOTES.starts_with(GENERATED_MARKER_PREFIX));
    }

    #[test]
    fn test_mcp_tools_json_is_valid_and_complete() {
        let tools: Vec<serde_json::Value> = serde_json::from_str(MCP_TOOLS_JSON).unwrap();
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert_eq!(
            names,
            ["send_message", "react", "schedule", "list_schedules", "cancel_schedule"]
        );
        for tool in &tools {
            assert!(tool["description"].is_string(), "{tool} has no description");
            assert!(tool["inputSchema"].is_object(), "{tool} has no inputSchema");
        }
        let send_message = tools.iter().find(|t| t["name"] == "send_message").unwrap();
        assert!(send_message["inputSchema"]["properties"]["file_path"].is_object());
    }

    /// The `tier` values the tool advertises are the ones `codex::tier` resolves.
    /// A schema offering a name `by_name` rejects turns every schedule creation
    /// into an error the agent cannot act on.
    #[test]
    fn test_schedule_tool_advertises_exactly_the_real_tiers() {
        let tools: Vec<serde_json::Value> = serde_json::from_str(MCP_TOOLS_JSON).unwrap();
        let schedule = tools.iter().find(|t| t["name"] == "schedule").unwrap();
        let advertised: Vec<&str> = schedule["inputSchema"]["properties"]["tier"]["enum"]
            .as_array()
            .expect("tier should be an enum")
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();

        let real: Vec<&str> = crate::codex::tier::ALL.iter().map(|t| t.name).collect();
        assert_eq!(advertised, real);
    }

    /// Voice rules the agent has to follow every turn. They have been lost to a
    /// rewrite once; this is the tripwire.
    #[test]
    fn test_the_root_instructions_still_ban_markdown() {
        assert!(WORKSPACE_AGENTS.contains("No markdown"));
    }

    #[test]
    fn test_the_root_instructions_keep_messages_informal() {
        assert!(WORKSPACE_AGENTS.contains("No em dashes, colons or semicolons in messages"));
        assert!(WORKSPACE_AGENTS.contains("informal, slightly goofy and witty"));
        assert!(WORKSPACE_AGENTS.contains("Absolutely, you're right"));
        assert!(WORKSPACE_AGENTS.contains("Never agree automatically"));
    }

    /// The tiers moved to WORKING.md when AGENTS.md was cut down. They have to be
    /// explained somewhere the agent will actually read before delegating.
    #[test]
    fn test_working_instructions_explain_every_model_tier() {
        assert!(WORKING.contains(crate::codex::tier::CONVERSATION.model));
        assert!(WORKING.contains(crate::codex::tier::HEAVY.model));
        for tier in crate::codex::tier::ALL {
            assert!(
                WORKING.contains(&format!("`{}`", tier.name)),
                "WORKING.md never explains the {:?} tier",
                tier.name
            );
        }
    }

    /// The point of the split is that AGENTS.md is read every session and the rest
    /// is not. If it drifts back to carrying everything, the saving is gone.
    #[test]
    fn test_the_root_instructions_stay_small_and_point_at_the_rest() {
        // Every session pays for this file. Keep the operational parts compact,
        // but leave enough room for the voice rules that make the agent distinct.
        let words = WORKSPACE_AGENTS.split_whitespace().count();
        assert!(
            words < 750,
            "AGENTS.md is {words} words; move detail into WORKING.md or a SCHEMA reference"
        );

        for pointer in [
            "WORKING.md",
            "SYSTEM.md",
            "history/SCHEMA.md",
            "logs/SCHEMA.md",
            "tasks/AGENTS.md",
            "projects/AGENTS.md",
        ] {
            assert!(
                WORKSPACE_AGENTS.contains(pointer),
                "AGENTS.md never tells the agent {pointer} exists"
            );
        }
    }

    #[test]
    fn test_the_root_instructions_teach_skill_timing_and_creation() {
        assert!(WORKSPACE_AGENTS.contains(".agents/skills"));
        assert!(WORKSPACE_AGENTS.contains("SKILL.md"));
        assert!(WORKSPACE_AGENTS.contains("nightly"));
        assert!(WORKSPACE_AGENTS.contains("100 characters"));
        assert!(!WORKSPACE_AGENTS.contains("Should I create a skill for this?"));
        assert!(WORKSPACE_AGENTS.contains("$skill-creator"));
        assert!(WORKING.contains("gpt-5.6-sol"));
        assert!(WORKING.contains("Create or improve a skill"));
        assert!(WORKING.contains("compact executable scripts"));
        assert!(WORKING.contains("Do not create `agents/openai.yaml`"));
    }

    #[test]
    fn test_the_root_instructions_pace_large_task_updates() {
        assert!(WORKSPACE_AGENTS.contains("use `send_message` while working"));
        assert!(WORKSPACE_AGENTS.contains("meaningful boundary"));
        assert!(WORKSPACE_AGENTS.contains("Do not narrate commands"));
        assert!(WORKSPACE_AGENTS.contains("load every detail at the front"));
        assert!(WORKSPACE_AGENTS.contains("large final message"));
    }

    #[test]
    fn test_markdown_prose_avoids_formal_punctuation() {
        for (name, text) in ALL {
            if !name.ends_with(".md") {
                continue;
            }
            for (line, prose) in markdown_prose(text) {
                for forbidden in [':', ';', '\u{2014}', '\u{2013}', '-'] {
                    assert!(
                        !prose.contains(forbidden),
                        "{name}:{line} contains {forbidden:?} in prose: {prose}"
                    );
                }
            }
        }
    }

    /// No em dashes anywhere in shipped text.
    ///
    /// Isala's rule, and a self-consistency one: the voice section tells the agent
    /// not to use them, and instructions that break their own rule are the weakest
    /// kind. Full stops and commas carry everything an em dash was doing.
    #[test]
    fn test_no_em_dashes_in_shipped_text() {
        for (name, text) in ALL {
            if let Some(at) = text.find('\u{2014}') {
                let around: String = text[at.saturating_sub(50)..(at + 50).min(text.len())].into();
                panic!("{name} contains an em dash: ...{around}...");
            }
        }
    }

    /// Prose must not be hard-wrapped: one paragraph, one line, and let the
    /// editor soft-wrap.
    ///
    /// A standing rule of Isala's, and it costs tokens for nothing. The check is
    /// the definition rather than a column guess: hard-wrapped prose is exactly a
    /// run of consecutive non-structural lines. Tables, list items, fenced code and
    /// the generated marker are structure, not prose, and neither is a short
    /// key/value line, which is why the run has to be of *long* lines to count.
    #[test]
    fn test_prose_is_not_hard_wrapped() {
        for (name, text) in ALL {
            if name.ends_with(".json") || name.ends_with(".toml") {
                continue;
            }

            let mut in_fence = false;
            let mut previous_was_prose = false;

            for (n, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("```") {
                    in_fence = !in_fence;
                    previous_was_prose = false;
                    continue;
                }

                let trimmed = line.trim_start();
                let is_prose = !in_fence
                    && !trimmed.is_empty()
                    && !trimmed.starts_with('|')
                    && !trimmed.starts_with('#')
                    && !trimmed.starts_with("- ")
                    && !trimmed.starts_with("<!--")
                    && !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit());

                // 60 chars: comfortably above a `Key: value` line, comfortably
                // below where a hard-wrapped paragraph breaks.
                let is_long_prose = is_prose && line.len() > 60;
                assert!(
                    !(is_long_prose && previous_was_prose),
                    "{name}:{} continues the previous line; unwrap the paragraph",
                    n + 1
                );
                previous_was_prose = is_long_prose;
            }
        }
    }
}
