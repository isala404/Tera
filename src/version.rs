use serde::{Deserialize, Serialize};
use std::process::Command;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const COMMIT_SHA: &str = env!("TERA_GIT_SHA");
pub const BUILD_TIME: &str = env!("TERA_BUILD_TIME");
pub const BUILD_TARGET: &str = env!("TERA_BUILD_TARGET");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    pub version: String,
    pub commit_sha: String,
    pub build_time: String,
    pub target: String,
}

impl BuildInfo {
    pub fn current() -> Self {
        Self {
            version: VERSION.to_string(),
            commit_sha: COMMIT_SHA.to_string(),
            build_time: BUILD_TIME.to_string(),
            target: BUILD_TARGET.to_string(),
        }
    }

    pub fn short_sha(&self) -> &str {
        self.commit_sha.get(..12).unwrap_or(&self.commit_sha)
    }
}

#[derive(Debug, Serialize)]
pub struct VersionReport {
    #[serde(flatten)]
    pub tera: BuildInfo,
    pub codex: Option<String>,
}

impl VersionReport {
    pub fn current() -> Self {
        Self {
            tera: BuildInfo::current(),
            codex: codex_version(),
        }
    }

    pub fn print(&self, json: bool) -> anyhow::Result<()> {
        if json {
            println!("{}", serde_json::to_string_pretty(self)?);
            return Ok(());
        }

        println!("Tera version:  {}", self.tera.version);
        println!("Commit SHA:    {}", self.tera.commit_sha);
        println!("Build time:    {}", self.tera.build_time);
        println!("Build target:  {}", self.tera.target);
        println!(
            "Codex version: {}",
            self.codex.as_deref().unwrap_or("not found")
        );
        Ok(())
    }
}

pub fn codex_version() -> Option<String> {
    let output = Command::new("codex").arg("--version").output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|version| !version.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_info_is_complete() {
        let info = BuildInfo::current();
        assert!(!info.version.is_empty());
        assert!(!info.commit_sha.is_empty());
        assert!(!info.build_time.is_empty());
        assert!(!info.target.is_empty());
        assert!(info.build_time.ends_with('Z'));
    }

    #[test]
    fn test_short_sha_handles_real_and_fallback_values() {
        let mut info = BuildInfo::current();
        info.commit_sha = "0123456789abcdef".to_string();
        assert_eq!(info.short_sha(), "0123456789ab");
        info.commit_sha = "unknown".to_string();
        assert_eq!(info.short_sha(), "unknown");
    }
}
