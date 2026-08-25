//! Workspace instruction files, rendered from the embedded `data/` templates.
//!
//! The text itself lives in `data/workspace/`; this module only fills in the
//! blanks. Ownership is split three ways so instructions can be improved without
//! destroying what someone else wrote:
//!
//! * files carrying the generated marker are ours. They are rewritten on every
//!   start, so a binary upgrade actually reaches the agent. Earlier versions were
//!   written only when absent, which meant a live workspace kept the first
//!   generation of instructions forever.
//! * `PERSONA.md` is the user's, and `SYSTEM.md` is the agent's own notebook about
//!   the machine. Both are created once and then never touched again.
//!
//! Every function takes the whole [`Config`] rather than the pieces it needs. The
//! templates interpolate a workspace path *and* an owner name, and taking them
//! separately makes it possible to render a file with the right path and a
//! forgotten owner, which ships a prompt still saying `{{OWNER}}` to the model.
//!
//! Nothing is hardcoded: not `/workspace`, because the workspace root is a flag and
//! instructions pointing at a directory that does not exist are worse than none;
//! and not a person's name, because this is a general tool and instructions
//! addressing somebody else's owner are worse still.

use crate::config::Config;
use crate::data;

pub use crate::data::GENERATED_MARKER_PREFIX;

pub fn render(template: &str, config: &Config) -> String {
    let vars = config.template_vars();
    let refs: Vec<(&str, &str)> = vars.iter().map(|(k, v)| (*k, v.as_str())).collect();
    data::render(template, &refs)
}

/// Seed the user-owned Codex model configuration.
pub fn generate_codex_config() -> String {
    data::CODEX_CONFIG_TOML.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A config with a fixed owner and binary, so assertions do not depend on the
    /// environment the tests happen to run in.
    fn config(workspace: &str) -> Config {
        let mut config = Config::new(PathBuf::from(workspace), true);
        config.owner_name = "Ada Lovelace".to_string();
        config.tera_bin = PathBuf::from("/usr/local/bin/tera");
        config
    }

    fn all_generated(config: &Config) -> Vec<String> {
        vec![
            render(data::WORKSPACE_AGENTS, config),
            render(data::CODEX_HOME_AGENTS, config),
            render(data::PROJECTS_AGENTS, config),
            render(data::TASKS_AGENTS, config),
            render(data::HISTORY_SCHEMA, config),
            render(data::LOGS_SCHEMA, config),
            render(data::SCHEDULE_AGENTS, config),
            render(data::WORKING, config),
        ]
    }

    #[test]
    fn test_templates_use_the_real_workspace_path() {
        let config = config("/tmp/my_workspace");
        let root = render(data::WORKSPACE_AGENTS, &config);
        let bootstrap = render(data::CODEX_HOME_AGENTS, &config);

        assert!(root.contains("/tmp/my_workspace/history/history.sqlite3"));
        assert!(root.contains("/tmp/my_workspace/WORKING.md"));
        assert!(bootstrap.contains("/tmp/my_workspace/AGENTS.md"));
        // The old templates hardcoded /workspace, which pointed Codex at a
        // directory that does not exist under any other workspace root.
        assert!(!root.contains("`/workspace"));
        assert!(!bootstrap.contains("`/workspace"));
    }

    /// The owner's name is configuration, not content: every file the agent reads
    /// has to address whoever is actually running it.
    #[test]
    fn test_every_rendered_file_addresses_the_configured_owner() {
        let config = config("/ws");
        let mut named = 0;

        for rendered in all_generated(&config).into_iter().chain([
            render(data::PERSONA, &config),
            render(data::SYSTEM_NOTES, &config),
        ]) {
            assert!(!rendered.contains("{{OWNER}}"), "unrendered owner name");
            if rendered.contains("Ada Lovelace") {
                named += 1;
            }
        }

        // The two schema references legitimately never mention the user; the files
        // that speak about them must.
        assert!(
            named >= 4,
            "only {named} rendered files addressed the owner"
        );
        assert!(render(data::WORKSPACE_AGENTS, &config).contains("Ada Lovelace"));
        assert!(render(data::PERSONA, &config).contains("Ada Lovelace"));
    }

    #[test]
    fn test_codex_config_uses_absolute_binary_path() {
        let cfg = config("/tmp/my_workspace").codex_overrides().join("\n");
        assert!(cfg.contains(r#"command="/usr/local/bin/tera""#));
        assert!(cfg.contains("/tmp/my_workspace/.runtime/assistant.sock"));
    }

    /// The MCP server is registered under one name and the instructions tell the
    /// agent to call tools on it by that name. If they drift, every tool call the
    /// agent tries is addressed to a server that does not exist.
    #[test]
    fn test_mcp_server_name_matches_everywhere() {
        let name = crate::mcp::stdio::MCP_SERVER_NAME;
        let config = config("/ws");
        let cfg = config.codex_overrides().join("\n");

        assert!(cfg.contains(&format!("mcp_servers.{name}.command")));
        assert!(render(data::WORKSPACE_AGENTS, &config).contains(&format!("`{name}` MCP server")));
        assert!(render(data::TASKS_AGENTS, &config).contains(&format!("`{name}` MCP server")));
    }

    /// A new workspace inherits native Codex defaults. The file is user-owned,
    /// so changing it through chat is configuration rather than a code change.
    #[test]
    fn test_codex_config_starts_with_native_defaults() {
        let cfg = generate_codex_config();
        assert!(cfg.contains("Leave it empty to use Codex's defaults"));
        assert!(!cfg.contains("model ="));
        assert!(!cfg.contains("model_provider ="));
        assert!(!cfg.contains("model_providers."));
    }

    /// A fresh Codex thread must not have to ask for the workspace, the disk or
    /// the network. Nobody is there to answer.
    #[test]
    fn test_codex_config_grants_full_privilege() {
        let cfg = config("/ws").codex_overrides().join("\n");
        assert!(cfg.contains(r#"approval_policy="never""#));
        assert!(cfg.contains(r#"sandbox_mode="danger-full-access""#));
        assert!(cfg.contains(r#"projects."/ws".trust_level="trusted""#));
        assert!(cfg.contains("sandbox_workspace_write.network_access=true"));
    }

    /// The agent is told to read history itself. If the instructions do not carry
    /// working commands, it falls back to guessing at paths.
    #[test]
    fn test_root_instructions_teach_the_shell_path_to_history() {
        let config = config("/ws");
        let root = render(data::WORKSPACE_AGENTS, &config);

        assert!(root.contains("/ws/history/jsonl/"));
        assert!(root.contains("jq"));
        assert!(root.contains("sqlite3 /ws/history/history.sqlite3"));
        assert!(root.contains("/ws/history/SCHEMA.md"));
        // The log reference lives in its own file now; AGENTS.md must point at it
        // rather than silently dropping the only way the agent diagnoses itself.
        assert!(root.contains("/ws/logs/SCHEMA.md"));
        assert!(render(data::LOGS_SCHEMA, &config).contains("tera::scheduler"));
        // There is no history tool any more; nothing may imply otherwise.
        assert!(!root.contains("history_search"));
    }

    /// The agent is told to read SYSTEM.md before touching the machine, and to
    /// keep it current. A skeleton with no prompts in it gets ignored.
    #[test]
    fn test_system_notes_are_a_fillable_skeleton_owned_by_the_agent() {
        let notes = render(data::SYSTEM_NOTES, &config("/ws"));
        assert!(!notes.starts_with(GENERATED_MARKER_PREFIX));
        assert!(notes.contains("Maintenance log"));
    }

    #[test]
    fn test_generated_files_are_marked_and_the_owned_ones_are_not() {
        let config = config("/ws");
        for generated in all_generated(&config) {
            assert!(
                generated.starts_with(GENERATED_MARKER_PREFIX),
                "{generated:.60}"
            );
        }
        // Neither of these is ours to rewrite, so neither may look like it is.
        assert!(!render(data::PERSONA, &config).starts_with(GENERATED_MARKER_PREFIX));
        assert!(!render(data::SYSTEM_NOTES, &config).starts_with(GENERATED_MARKER_PREFIX));
    }

    /// Nothing may reach a workspace file with a placeholder still in it.
    #[test]
    fn test_rendered_instructions_have_no_leftover_placeholders() {
        let config = config("/ws");
        for rendered in all_generated(&config).into_iter().chain([
            render(data::PERSONA, &config),
            render(data::SYSTEM_NOTES, &config),
            generate_codex_config(),
        ]) {
            assert!(!rendered.contains("{{"), "{rendered:.200}");
        }
    }
}
