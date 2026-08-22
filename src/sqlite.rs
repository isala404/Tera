//! The one SQLite chore both databases need.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use tracing::info;

/// Add a column to a table that already exists in a live workspace.
///
/// `CREATE TABLE IF NOT EXISTS` never revisits a table it did not create, so a
/// new column in a schema constant reaches a fresh database and no other. SQLite
/// has no `ADD COLUMN IF NOT EXISTS`, so every column added after a release
/// needs a call to this.
pub fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    decl: &str,
) -> Result<()> {
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM pragma_table_info(?1) WHERE name = ?2",
        params![table, column],
        |row| row.get(0),
    )?;
    if exists {
        return Ok(());
    }

    info!("Migrating {table}: adding column {column}");
    conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))
        .with_context(|| format!("Failed to add {table}.{column}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_missing_column_is_added_and_a_present_one_is_left_alone() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (a TEXT);").unwrap();

        add_column_if_missing(&conn, "t", "b", "INTEGER NOT NULL DEFAULT 0").unwrap();
        conn.execute("INSERT INTO t (a, b) VALUES ('x', 7)", [])
            .unwrap();

        // The second call must not clobber the value the first one made room for.
        add_column_if_missing(&conn, "t", "b", "INTEGER NOT NULL DEFAULT 0").unwrap();
        let b: i64 = conn
            .query_row("SELECT b FROM t", [], |row| row.get(0))
            .unwrap();
        assert_eq!(b, 7);
    }
}
