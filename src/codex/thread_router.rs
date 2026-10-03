//! Thread-selection policy for the main conversation.
//!
//! This decides *whether* to keep talking on the current thread. It deliberately
//! does not mint thread ids: only the app-server can, and an earlier version that
//! generated its own `th_<uuid>` produced ids no `thread/resume` would ever
//! accept.

use crate::config::Config;
use crate::runtime::MainThreadState;
use std::fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadDecision {
    /// Keep the conversation where it is; resume it first if it is not loaded.
    Continue { thread_id: String },
    /// Start a fresh thread. The prompt cache is cold, so there is nothing left
    /// to reuse.
    Rotate { reason: String },
}

pub struct ThreadRouter;

impl ThreadRouter {
    /// A model change needs no rotation of its own: resuming applies the
    /// current model to the thread and keeps its context.
    pub fn decide(persisted: Option<&MainThreadState>, now_ms: i64) -> ThreadDecision {
        let Some(state) = persisted else {
            return ThreadDecision::Rotate {
                reason: "no conversation thread recorded yet".to_string(),
            };
        };

        if now_ms >= state.estimated_cache_warm_until_ms {
            return ThreadDecision::Rotate {
                reason: format!(
                    "thread {} has been idle past its cache window",
                    state.thread_id
                ),
            };
        }

        ThreadDecision::Continue {
            thread_id: state.thread_id.clone(),
        }
    }

    /// Pointers handed to a thread that starts with nothing. The caller adds
    /// the recent conversation itself.
    ///
    /// Kept deliberately small. A fresh thread gets pointers, never a
    /// synthesized context blob. `AGENTS.md` tells
    /// the agent to read `HORIZON.md` and `INDEX.md` itself, and it reads them
    /// better than we can summarize them.
    pub fn build_bootstrap_context(config: &Config) -> String {
        let mut context = format!(
            "This is a fresh thread, so your earlier context is gone but the conversation \
             is not. Before replying, read the recent conversation below the way a person \
             rejoining a chat would. Work out what {owner} is after, what was already done \
             and what you already told them, then answer the newest message from where \
             things actually stand. Never repeat something you already said, redo finished \
             work or reopen a settled topic. Follow the conversation, not a checklist.\n\n\
             Then read these:\n",
            owner = config.owner_name
        );

        context.push_str(&format!("- {}\n", config.root_agents_path().display()));
        if config.persona_path().exists() {
            context.push_str(&format!("- {}\n", config.persona_path().display()));
        }

        let memories = config.memories_dir();
        for file in ["HORIZON.md", "INDEX.md"] {
            if memories.join(file).exists() {
                context.push_str(&format!("- {}\n", memories.join(file).display()));
            }
        }

        let jsonl = config.history_jsonl_dir();
        if fs::read_dir(&jsonl)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false)
        {
            context.push_str(&format!(
                "\nIf the recent conversation doesn't show where things stand, look further back with `cat {}/*.jsonl | tail -60 | jq -c .`\n",
                jsonl.display()
            ));
        }

        context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codex::CACHE_TTL_MS as TTL;
    const NOW: i64 = 1_786_962_664_000;

    fn persisted(warm_until_ms: i64) -> MainThreadState {
        MainThreadState {
            thread_id: "thread_real".to_string(),
            started_at_ms: NOW - TTL,
            last_activity_at_ms: warm_until_ms - TTL,
            estimated_cache_warm_until_ms: warm_until_ms,
            model_id: "model-a".to_string(),
        }
    }

    #[test]
    fn test_first_ever_turn_starts_a_thread() {
        assert!(matches!(
            ThreadRouter::decide(None, NOW),
            ThreadDecision::Rotate { .. }
        ));
    }

    #[test]
    fn test_warm_persisted_thread_is_continued() {
        let decision = ThreadRouter::decide(Some(&persisted(NOW + TTL)), NOW);
        assert_eq!(
            decision,
            ThreadDecision::Continue {
                thread_id: "thread_real".to_string()
            }
        );
    }

    /// Including a thread that is still loaded in this process: the replacement
    /// is handed recent canonical history, so rotating no longer drops the
    /// conversation the user can still see.
    #[test]
    fn test_cold_thread_is_rotated() {
        let decision = ThreadRouter::decide(Some(&persisted(NOW - 1)), NOW);
        assert!(matches!(decision, ThreadDecision::Rotate { .. }));
    }
}
