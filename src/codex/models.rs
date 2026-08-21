use crate::codex::tier;
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
    /// Record which model this daemon runs on, and flag a memory rebuild when it
    /// changes.
    ///
    /// The tracked model is *ours*, [`tier::CONVERSATION`], not the app-server's
    /// advertised default. The rebuild exists because memory is one model's
    /// interpretation of history and a different model may organise it differently,
    /// so the model that matters is the one actually writing it. Watching the
    /// vendor's default instead meant an expensive full rebuild every time OpenAI
    /// promoted a new frontier model tera does not even use.
    ///
    /// The model list is still worth fetching: it is how we learn our pinned model
    /// has been withdrawn, which is otherwise a turn failure with no explanation.
    pub fn process_models_response(
        runtime_db: &RuntimeDb,
        models: Vec<ModelDescriptor>,
    ) -> Result<Option<ModelDescriptor>> {
        let ours = tier::CONVERSATION.model;

        if !models.is_empty() && !models.iter().any(|m| m.id == ours) {
            warn!(
                "Configured model {ours} is not in the app-server's list ({}).                  Turns will fail until codex::tier is updated.",
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
        let changed = matches!(&last, Some(prev) if prev.model_id != ours);
        if changed {
            info!(
                "Model tera runs on changed from {:?} to {ours}. Flagging rebuild_pending!",
                last.as_ref().map(|m| &m.model_id)
            );
            runtime_db.set_state_value("rebuild_pending", "true")?;
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
                id: "gpt-5.6-sol".into(),
                name: Some("GPT-5.6-Sol".into()),
                is_default: true,
            },
            ModelDescriptor {
                id: tier::CONVERSATION.model.into(),
                name: Some("GPT-5.6-Luna".into()),
                is_default: false,
            },
        ]
    }

    /// The whole point of the change: the vendor's default is Sol and always has
    /// been, and it changing must not trigger a full memory rebuild on a daemon
    /// that pins Luna.
    #[test]
    fn test_a_vendor_default_change_does_not_trigger_a_rebuild() {
        let runtime_db = db();
        ModelDiscovery::process_models_response(&runtime_db, listing()).unwrap();

        let mut promoted = listing();
        promoted[0].id = "gpt-5.7-sol".into();
        ModelDiscovery::process_models_response(&runtime_db, promoted).unwrap();

        assert_eq!(runtime_db.get_state_value("rebuild_pending").unwrap(), None);
    }

    #[test]
    fn test_it_records_the_model_we_actually_run() {
        let runtime_db = db();
        ModelDiscovery::process_models_response(&runtime_db, listing()).unwrap();

        let recorded = runtime_db.get_last_default_model().unwrap().unwrap();
        assert_eq!(recorded.model_id, tier::CONVERSATION.model);
    }

    /// Changing `codex::tier` is the case the rebuild is for: memory was written by
    /// the old model and the new one may want it organised differently.
    #[test]
    fn test_changing_our_own_model_triggers_a_rebuild() {
        let runtime_db = db();
        runtime_db
            .record_model_observation(&ModelObservation {
                model_id: "gpt-5.4-mini".into(),
                display_name: None,
                is_default: true,
                observed_at_ms: 0,
            })
            .unwrap();

        ModelDiscovery::process_models_response(&runtime_db, listing()).unwrap();
        assert_eq!(
            runtime_db.get_state_value("rebuild_pending").unwrap().as_deref(),
            Some("true")
        );
    }

    /// A pinned model that has been withdrawn is a turn failure with no visible
    /// cause; the first run is the only chance to say so.
    #[test]
    fn test_a_missing_pinned_model_still_records_and_does_not_panic() {
        let runtime_db = db();
        let without_ours = vec![ModelDescriptor {
            id: "gpt-5.6-sol".into(),
            name: None,
            is_default: true,
        }];

        let found = ModelDiscovery::process_models_response(&runtime_db, without_ours).unwrap();
        assert!(found.is_none());
        assert_eq!(
            runtime_db.get_last_default_model().unwrap().unwrap().model_id,
            tier::CONVERSATION.model
        );
    }
}
