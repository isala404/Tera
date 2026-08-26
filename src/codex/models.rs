use crate::runtime::{ModelObservation, RuntimeDb};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub name: Option<String>,
    #[serde(default)]
    pub is_default: bool,
}

pub struct ModelDiscovery;

impl ModelDiscovery {
    /// Record which model this daemon runs on, and say so when it changes.
    ///
    /// The tracked model is the one Codex resolved from Tera's user-owned config,
    /// not the app-server's advertised default.
    ///
    /// A change is worth noticing but not worth acting on. Memory is one model's
    /// interpretation of history, so a new model may well organise it better, but
    /// re-deriving the whole tree costs millions of tokens and switching models is
    /// a thing an owner does casually, several times an evening. Rebuilding on
    /// that signal made a config edit cost more than a month of conversation. The
    /// owner decides, by asking for a rebuild.
    ///
    /// The model list is still worth fetching: it is how we learn our pinned model
    /// has been withdrawn, which is otherwise a turn failure with no explanation.
    pub fn process_models_response(
        runtime_db: &RuntimeDb,
        configured_model: &str,
        models: Vec<ModelDescriptor>,
    ) -> Result<Option<ModelDescriptor>> {
        let ours = configured_model;

        if !models.is_empty() && !models.iter().any(|m| m.id == ours) {
            warn!(
                "Configured model {ours} is not in the app-server's list ({}). Check .codex-home/config.toml and the provider's model catalog.",
                models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ")
            );
        }

        if let Some(vendor_default) = models.iter().find(|m| m.is_default) {
            info!(
                "App-server default model is {} ({:?}); tera runs {ours}",
                vendor_default.id, vendor_default.name
            );
        }

        let last = runtime_db.get_last_default_model()?;
        if let Some(prev) = last.as_ref().filter(|prev| prev.model_id != ours) {
            info!(
                "Model tera runs on changed from {} to {ours}. Memory still reflects \
                 the old model; ask for a memory rebuild if you want it re-derived",
                prev.model_id
            );
        }

        runtime_db.record_model_observation(&ModelObservation {
            model_id: ours.to_string(),
            display_name: models
                .iter()
                .find(|m| m.id == ours)
                .and_then(|m| m.name.clone()),
            is_default: true,
            observed_at_ms: Utc::now().timestamp_millis(),
        })?;

        Ok(models.into_iter().find(|m| m.id == ours))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> RuntimeDb {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.sqlite3");
        std::mem::forget(dir);
        RuntimeDb::open(&path).unwrap()
    }

    fn listing() -> Vec<ModelDescriptor> {
        vec![
            ModelDescriptor {
                id: "vendor-default".into(),
                name: Some("Vendor default".into()),
                is_default: true,
            },
            ModelDescriptor {
                id: "configured-model".into(),
                name: Some("Configured conversation model".into()),
                is_default: false,
            },
        ]
    }

    fn configured() -> &'static str {
        "configured-model"
    }

    /// A vendor default changing must not disturb a daemon that pins another
    /// model: the model we run is the one we record.
    #[test]
    fn test_a_vendor_default_change_does_not_change_what_we_record() {
        let runtime_db = db();
        ModelDiscovery::process_models_response(&runtime_db, configured(), listing()).unwrap();

        let mut promoted = listing();
        promoted[0].id = "new-vendor-default".into();
        ModelDiscovery::process_models_response(&runtime_db, configured(), promoted).unwrap();

        let recorded = runtime_db.get_last_default_model().unwrap().unwrap();
        assert_eq!(recorded.model_id, configured());
    }

    #[test]
    fn test_it_records_the_model_we_actually_run() {
        let runtime_db = db();
        ModelDiscovery::process_models_response(&runtime_db, configured(), listing()).unwrap();

        let recorded = runtime_db.get_last_default_model().unwrap().unwrap();
        assert_eq!(recorded.model_id, configured());
    }

    /// Switching the configured model records the new one and nothing else. A
    /// rebuild costs millions of tokens, and an owner trying providers out
    /// switches models several times an evening.
    #[test]
    fn test_changing_our_own_model_only_records_it() {
        let runtime_db = db();
        runtime_db
            .record_model_observation(&ModelObservation {
                model_id: "previous-model".into(),
                display_name: None,
                is_default: true,
                observed_at_ms: 0,
            })
            .unwrap();

        ModelDiscovery::process_models_response(&runtime_db, configured(), listing()).unwrap();

        assert_eq!(
            runtime_db
                .get_last_default_model()
                .unwrap()
                .unwrap()
                .model_id,
            configured()
        );
    }

    /// A pinned model that has been withdrawn is a turn failure with no visible
    /// cause; the first run is the only chance to say so.
    #[test]
    fn test_a_missing_pinned_model_still_records_and_does_not_panic() {
        let runtime_db = db();
        let without_ours = vec![ModelDescriptor {
            id: "vendor-default".into(),
            name: None,
            is_default: true,
        }];

        let found =
            ModelDiscovery::process_models_response(&runtime_db, configured(), without_ours)
                .unwrap();
        assert!(found.is_none());
        assert_eq!(
            runtime_db
                .get_last_default_model()
                .unwrap()
                .unwrap()
                .model_id,
            configured()
        );
    }
}
