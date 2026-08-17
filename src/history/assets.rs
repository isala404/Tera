use crate::config::Config;
use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use std::fs;
use std::path::PathBuf;

pub struct AssetStorage;

impl AssetStorage {
    pub fn save_attachment(
        config: &Config,
        message_id: &str,
        occurred_at_ms: i64,
        filename: &str,
        data: &[u8],
    ) -> Result<(PathBuf, String)> {
        let dt: DateTime<Utc> = Utc.timestamp_millis_opt(occurred_at_ms).unwrap();
        let year_str = dt.format("%Y").to_string();
        let month_str = dt.format("%m").to_string();

        let asset_dir = config
            .history_assets_dir()
            .join(&year_str)
            .join(&month_str)
            .join(message_id);

        fs::create_dir_all(&asset_dir)?;
        let full_path = asset_dir.join(filename);
        fs::write(&full_path, data)
            .with_context(|| format!("Failed to write media asset to {:?}", full_path))?;

        let relative_path = format!(
            "../assets/{}/{}/{}/{}",
            year_str, month_str, message_id, filename
        );

        Ok((full_path, relative_path))
    }
}
