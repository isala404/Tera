//! Database setup utilities for testing with pgvector
//!
//! pgvector is REQUIRED for this application. All test databases must have
//! the pgvector extension installed and enabled.

#[cfg(test)]
pub mod setup {
    use forge::testing::{IsolatedTestDb, TestDatabase};
    use std::path::Path;

    pub fn is_pgvector_unavailable(err: &(dyn std::error::Error + 'static)) -> bool {
        let msg = err.to_string();
        msg.contains("pgvector extension is REQUIRED")
            || msg.contains("extension \"vector\" is not available")
            || msg.contains("could not open extension control file")
    }

    /// Initialize a test database with pgvector extension (REQUIRED)
    ///
    /// pgvector must be available in the PostgreSQL environment.
    /// For embedded tests, ensure pgvector is installed.
    pub async fn init_test_db_with_vector(
        name: &str,
    ) -> Result<IsolatedTestDb, Box<dyn std::error::Error>> {
        // Create embedded test database
        let base = TestDatabase::embedded().await?;
        let db = base.isolated(name).await?;

        // Run internal Forge SQL setup
        db.run_sql(&forge::get_internal_sql()).await?;

        // Enable pgvector extension - REQUIRED, no fallback
        let pool = db.pool();
        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(pool)
            .await
            .map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "pgvector extension is REQUIRED but not available: {}. \
                        Ensure PostgreSQL is compiled with pgvector support.",
                        e
                    ),
                )) as Box<dyn std::error::Error>
            })?;

        // Run migrations (now requires pgvector to be available)
        db.migrate(Path::new("migrations")).await?;

        Ok(db)
    }
}

#[cfg(test)]
mod tests {
    use super::setup;

    #[tokio::test]
    async fn test_db_init_with_pgvector() {
        match setup::init_test_db_with_vector("test_pgvector").await {
            Ok(_) => {}
            Err(err) if setup::is_pgvector_unavailable(err.as_ref()) => {
                eprintln!("Skipping pgvector test: {}", err);
            }
            Err(err) => panic!("Unexpected test DB init failure: {}", err),
        }
    }
}
