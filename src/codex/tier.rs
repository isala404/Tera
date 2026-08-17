//! Which model runs a turn, and how hard it thinks.
//!
//! Cheap by default, expensive only when the thinking is genuinely hard. Left to
//! itself the app-server picks its own frontier default for everything, which
//! spends Sol-grade reasoning on a disk check.
//!
//! The tiers are named rather than passed as raw model/effort pairs so the
//! decision is made once, here, instead of at every call site, and so the
//! `schedule` tool can take a word the agent understands instead of asking it to
//! remember model ids.

use anyhow::{anyhow, Result};

/// A model and a reasoning effort, addressed by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelTier {
    /// The value stored on a schedule and accepted by the `schedule` tool.
    pub name: &'static str,
    pub model: &'static str,
    /// A `ReasoningEffort` the model advertises: low, medium, high, xhigh, max.
    pub effort: &'static str,
}

/// Mechanical work of known shape: checks, sweeps, extracting numbers from a log,
/// a brief assembled from sources that are already identified.
pub const ROUTINE: ModelTier = ModelTier {
    name: "routine",
    model: "gpt-5.6-luna",
    effort: "low",
};

/// Everything ordinary, and every conversation turn.
pub const CONVERSATION: ModelTier = ModelTier {
    name: "default",
    model: "gpt-5.6-luna",
    effort: "xhigh",
};

/// Work that is hard rather than merely important: a cause that is not obvious,
/// an approach that has to be designed, many interacting constraints. Also what the
/// owner means when they ask for "sol" by name.
pub const HEAVY: ModelTier = ModelTier {
    name: "heavy",
    model: "gpt-5.6-sol",
    effort: "high",
};

pub const ALL: [ModelTier; 3] = [ROUTINE, CONVERSATION, HEAVY];

/// Resolve a tier name from a tool argument or a stored schedule.
///
/// An unknown name is an error rather than a silent fall back to the default: a
/// typo that quietly downgrades a heavy task is invisible, and the agent can read
/// the message and retry.
pub fn by_name(name: &str) -> Result<ModelTier> {
    ALL.iter().find(|t| t.name == name).copied().ok_or_else(|| {
        anyhow!(
            "unknown tier {name:?}; use one of: {}",
            ALL.iter().map(|t| t.name).collect::<Vec<_>>().join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiers_resolve_by_name() {
        assert_eq!(by_name("routine").unwrap(), ROUTINE);
        assert_eq!(by_name("default").unwrap(), CONVERSATION);
        assert_eq!(by_name("heavy").unwrap(), HEAVY);
    }

    /// Silently substituting the default for a typo would downgrade a heavy task
    /// with nothing in the log to show it happened.
    #[test]
    fn test_an_unknown_tier_is_an_error_naming_the_valid_ones() {
        let err = by_name("expensive").unwrap_err().to_string();
        assert!(err.contains("routine"), "unhelpful error: {err}");
        assert!(err.contains("heavy"), "unhelpful error: {err}");
    }

    /// The tool schema the agent reads names these efforts and models; a tier
    /// whose effort the model does not advertise is rejected by the app-server at
    /// turn/start, which looks like an unexplained turn failure.
    #[test]
    fn test_every_tier_uses_an_advertised_effort() {
        for tier in ALL {
            assert!(
                ["low", "medium", "high", "xhigh", "max"].contains(&tier.effort),
                "{} uses effort {:?}",
                tier.name,
                tier.effort
            );
            assert!(tier.model.starts_with("gpt-"), "{:?}", tier.model);
        }
    }
}
