//! Thread-selection policy for the main conversation (PLAN.md section 12.3).
//!
//! This decides *whether* to keep talking on the current thread. It deliberately
//! does not mint thread ids: only the app-server can, and an earlier version that
//! generated its own `th_<uuid>` produced ids no `thread/resume` would ever
//! accept.

use crate::config::Config;
use crate::runtime::RuntimeDb;
use anyhow::Result;
use chrono::Utc;
use std::fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadDecision {
    /// Keep the conversation where it is; resume it first if it is not loaded.
    Continue { thread_id: String },
    /// Start a fresh thread. The prompt cache is cold or the model changed, so
    /// there is nothing left to reuse.
    Rotate { reason: String },
}

pub struct ThreadRouter;

impl ThreadRouter {
    pub fn decide(
        config: &Config,
        runtime_db: &RuntimeDb,
        current_model_id: &str,
    ) -> Result<ThreadDecision> {
        Ok(Self::decide_at(
            runtime_db.get_main_thread()?.as_ref().map(|s| PersistedThread {
                thread_id: s.thread_id.clone(),
                estimated_cache_warm_until_ms: s.estimated_cache_warm_until_ms,
                model_id: s.model_id.clone(),
            }),
            current_model_id,
            Utc::now().timestamp_millis(),
            config.cache_ttl_ms(),
        ))
    }

    /// The policy itself, with time and state passed in so it is testable.
    fn decide_at(
        persisted: Option<PersistedThread>,
        current_model_id: &str,
        now_ms: i64,
        _ttl_ms: i64,
    ) -> ThreadDecision {
        let Some(state) = persisted else {
            return ThreadDecision::Rotate {
                reason: "no conversation thread recorded yet".to_string(),
            };
        };

        if !current_model_id.is_empty() && state.model_id != current_model_id {
            return ThreadDecision::Rotate {
                reason: format!(
                    "model changed from {} to {current_model_id}",
                    state.model_id
                ),
            };
        }

        if now_ms >= state.estimated_cache_warm_until_ms {
            return ThreadDecision::Rotate {
                reason: format!(
                    "thread {} has been idle past its cache window",
                    state.thread_id
                ),
            };
        }

        ThreadDecision::Continue {
            thread_id: state.thread_id,
        }
    }

    /// Pointers handed to a thread that starts with nothing. The caller adds
    /// the recent conversation itself.
    ///
    /// Kept deliberately small. PLAN.md section 12.4 is explicit that a fresh
    /// thread should not receive a synthesized context blob. `AGENTS.md` tells
    /// the agent to read `HORIZON.md` and `INDEX.md` itself, and it reads them
    /// better than we can summarize them.
    pub fn build_bootstrap_context(config: &Config) -> String {
        let mut context = String::from(
            "This is a fresh conversation thread; the earlier context is not in \
             your window. Before answering, read these:\n",
        );

        context.push_str(&format!("- {}\n", config.root_agents_path().display()));
        if config.persona_path().exists() {
            context.push_str(&format!("- {}\n", config.persona_path().display()));
        }

        let memories = config.memories_link();
        for file in ["HORIZON.md", "INDEX.md"] {
            if memories.join(file).exists() {
                context.push_str(&format!("- {}\n", memories.join(file).display()));
            }
        }

        let jsonl = config.history_jsonl_dir();
        if fs::read_dir(&jsonl).map(|mut d| d.next().is_some()).unwrap_or(false) {
            context.push_str(&format!(
                "\nRecent conversation, if you need it: `cat {}/*.jsonl | tail -20 | jq -c .`\n",
                jsonl.display()
            ));
        }

        context
    }
}

#[derive(Debug, Clone)]
struct PersistedThread {
    thread_id: String,
    estimated_cache_warm_until_ms: i64,
    model_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: i64 = 30 * 60 * 1000;
    const NOW: i64 = 1_786_962_664_000;

    fn persisted(warm_until_ms: i64, model: &str) -> Option<PersistedThread> {
        Some(PersistedThread {
            thread_id: "thread_real".to_string(),
            estimated_cache_warm_until_ms: warm_until_ms,
            model_id: model.to_string(),
        })
    }

    #[test]
    fn test_first_ever_turn_starts_a_thread() {
        assert!(matches!(
            ThreadRouter::decide_at(None, "gpt-5.6-sol", NOW, TTL),
            ThreadDecision::Rotate { .. }
        ));
    }

    #[test]
    fn test_warm_persisted_thread_is_continued() {
        let decision =
            ThreadRouter::decide_at(persisted(NOW + TTL, "gpt-5.6-sol"), "gpt-5.6-sol", NOW, TTL);
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
        let decision =
            ThreadRouter::decide_at(persisted(NOW - 1, "gpt-5.6-sol"), "gpt-5.6-sol", NOW, TTL);
        assert!(matches!(decision, ThreadDecision::Rotate { .. }));
    }

    #[test]
    fn test_model_change_rotates_even_when_warm() {
        let decision =
            ThreadRouter::decide_at(persisted(NOW + TTL, "gpt-5.6-sol"), "gpt-6", NOW, TTL);
        match decision {
            ThreadDecision::Rotate { reason } => assert!(reason.contains("gpt-6"), "{reason}"),
            other => panic!("expected rotation, got {other:?}"),
        }
    }

    /// An unknown current model is not evidence of a change.
    #[test]
    fn test_unknown_model_does_not_force_rotation() {
        let decision =
            ThreadRouter::decide_at(persisted(NOW + TTL, "gpt-5.6-sol"), "", NOW, TTL);
        assert!(matches!(decision, ThreadDecision::Continue { .. }));
    }
}
