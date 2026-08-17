use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Used when neither `TERA_OWNER` nor the login name is available.
///
/// Reads correctly in a sentence, which matters because it goes straight into
/// prompts: "the personal assistant for the owner" is merely impersonal, where a
/// leftover `{{OWNER}}` or an empty string is broken.
const UNKNOWN_OWNER: &str = "the owner";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub workspace_dir: PathBuf,
    /// What to call the person this assistant works for, in every prompt and
    /// instruction file.
    ///
    /// Templated rather than written into the prompts because this is a general
    /// tool, not one person's: a name in `data/` would make every other user's
    /// assistant address them as someone else. Set `TERA_OWNER`.
    pub owner_name: String,
    pub whatsapp_owner_number: Option<String>,
    pub mock_transport: bool,
    pub cache_ttl_minutes: u64,
    /// Absolute path to this binary, written into the Codex `config.toml` so
    /// Codex can spawn the MCP proxy. Overridable via `TERA_BIN`. Under
    /// `cargo test` the current executable is the test harness, not the daemon.
    pub tera_bin: PathBuf,
}

impl Config {
    pub fn new(workspace_dir: PathBuf, mock_transport: bool) -> Self {
        // Accept either name: the owner may be identified by phone number or by
        // the `@lid` identifier WhatsApp now uses for linked devices.
        let whatsapp_owner_number = std::env::var("WHATSAPP_OWNER_JID")
            .or_else(|_| std::env::var("WHATSAPP_OWNER_NUMBER"))
            .ok();

        let tera_bin = std::env::var_os("TERA_BIN")
            .map(PathBuf::from)
            .or_else(|| std::env::current_exe().ok())
            .unwrap_or_else(|| PathBuf::from("tera"));

        Self {
            workspace_dir,
            owner_name: resolve_owner_name(),
            whatsapp_owner_number,
            mock_transport,
            cache_ttl_minutes: 30,
            tera_bin,
        }
    }

    /// Prompt-cache lifetime, as milliseconds. An estimate, not a fact: the
    /// app-server exposes no per-thread cache expiry (PLAN.md section 12.2).
    pub fn cache_ttl_ms(&self) -> i64 {
        (self.cache_ttl_minutes * 60 * 1000) as i64
    }

    pub fn runtime_dir(&self) -> PathBuf {
        self.workspace_dir.join(".runtime")
    }

    pub fn socket_path(&self) -> PathBuf {
        self.runtime_dir().join("assistant.sock")
    }

    pub fn lock_file_path(&self) -> PathBuf {
        self.runtime_dir().join("locks").join("daemon.lock")
    }

    pub fn history_db_path(&self) -> PathBuf {
        self.workspace_dir.join("history").join("history.sqlite3")
    }

    pub fn history_jsonl_dir(&self) -> PathBuf {
        self.workspace_dir.join("history").join("jsonl")
    }

    pub fn history_assets_dir(&self) -> PathBuf {
        self.workspace_dir.join("history").join("assets")
    }

    pub fn runtime_db_path(&self) -> PathBuf {
        self.runtime_dir().join("state.sqlite3")
    }

    /// The daemon's own log. Not under `.runtime/`: the agent is told to read it
    /// when something looks broken, and a hidden directory is a poor place to
    /// point somebody at.
    pub fn logs_dir(&self) -> PathBuf {
        self.workspace_dir.join("logs")
    }

    pub fn codex_home_dir(&self) -> PathBuf {
        self.workspace_dir.join(".codex-home")
    }

    /// The active memory generation, reached through a symlink.
    ///
    /// Uppercase like every other knowledge file in the workspace. macOS is
    /// case-insensitive so a workspace created under the old lowercase name needs
    /// no migration there; `WorkspaceInit` removes the stale link on Linux.
    pub fn memories_link(&self) -> PathBuf {
        self.workspace_dir.join("MEMORIES")
    }

    /// The pre-1.2 lowercase name, kept only so init can clean it up.
    pub fn legacy_memories_link(&self) -> PathBuf {
        self.workspace_dir.join("memories")
    }

    pub fn generations_dir(&self) -> PathBuf {
        self.workspace_dir.join(".memory").join("generations")
    }

    pub fn staging_dir(&self) -> PathBuf {
        self.workspace_dir.join(".memory").join("staging")
    }

    pub fn projects_dir(&self) -> PathBuf {
        self.workspace_dir.join("projects")
    }

    pub fn tasks_dir(&self) -> PathBuf {
        self.workspace_dir.join("tasks")
    }

    pub fn root_agents_path(&self) -> PathBuf {
        self.workspace_dir.join("AGENTS.md")
    }

    /// The user's own instructions. Created once, never rewritten.
    pub fn persona_path(&self) -> PathBuf {
        self.workspace_dir.join("PERSONA.md")
    }

    /// The agent's notebook about the machine it runs on. Seeded once with a
    /// skeleton and then owned entirely by the agent, nothing here rewrites it,
    /// and the memory maintenance passes are told to leave it alone. It holds what
    /// is true of the host rather than of the user, which is not derivable from
    /// conversation history and so cannot live in a memory generation.
    pub fn system_notes_path(&self) -> PathBuf {
        self.workspace_dir.join("SYSTEM.md")
    }

    /// The substitutions every instruction file and prompt needs.
    ///
    /// One place, so a new template cannot be rendered with the workspace path but
    /// no owner name, which would ship a file still saying `{{OWNER}}` to the
    /// model.
    pub fn template_vars(&self) -> Vec<(&'static str, String)> {
        vec![
            ("WORKSPACE", self.workspace_dir.display().to_string()),
            ("OWNER", self.owner_name.clone()),
        ]
    }

    /// Turn a stored attachment path back into an absolute one.
    ///
    /// Attachment paths are stored relative to `history/jsonl/` so the JSONL
    /// projection can link to them, which makes them meaningless to anything
    /// else without this resolution step.
    pub fn resolve_asset(&self, relative_path: &str) -> PathBuf {
        let joined = self.history_jsonl_dir().join(relative_path);
        joined.canonicalize().unwrap_or(joined)
    }
}

/// Who the assistant works for: `TERA_OWNER`, else the login name, else a generic
/// stand-in.
///
/// Environment rather than a flag or a workspace file, for the same reason
/// `WHATSAPP_OWNER_JID` is: the daemon is started by a service unit or a shell
/// line that can carry it, and adding persisted state means a second place for the
/// answer to be wrong. Falling back to the login name is not a guess, on a
/// personal machine the account holder *is* the owner, but the resolved value is
/// logged at startup and printed by `tera status`, because a forgotten `TERA_OWNER`
/// would otherwise silently change who the agent thinks it works for.
fn resolve_owner_name() -> String {
    std::env::var("TERA_OWNER")
        .ok()
        .or_else(|| std::env::var("USER").ok())
        .or_else(|| std::env::var("LOGNAME").ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| UNKNOWN_OWNER.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blank `TERA_OWNER` is a misconfiguration, not an instruction to address
    /// the user as "". It must fall through to something that reads as a name.
    #[test]
    fn test_owner_name_is_never_empty() {
        let cfg = Config::new(PathBuf::from("/ws"), true);
        assert!(!cfg.owner_name.trim().is_empty());

        let vars = cfg.template_vars();
        let owner = vars.iter().find(|(k, _)| *k == "OWNER").unwrap();
        assert!(!owner.1.is_empty());
    }

    #[test]
    fn test_config_paths() {
        let root = PathBuf::from("/tmp/test_workspace");
        let cfg = Config::new(root.clone(), true);

        assert_eq!(cfg.runtime_dir(), root.join(".runtime"));
        assert_eq!(cfg.socket_path(), root.join(".runtime/assistant.sock"));
        assert_eq!(cfg.history_db_path(), root.join("history/history.sqlite3"));
        assert_eq!(cfg.runtime_db_path(), root.join(".runtime/state.sqlite3"));
        assert_eq!(cfg.codex_home_dir(), root.join(".codex-home"));
        assert_eq!(cfg.memories_link(), root.join("MEMORIES"));
        assert_eq!(cfg.system_notes_path(), root.join("SYSTEM.md"));
        assert_eq!(cfg.generations_dir(), root.join(".memory/generations"));
    }
}
