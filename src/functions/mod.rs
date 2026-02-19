// Define your functions here
// Example:
// pub mod users;
// pub use users::*;

// Example test module - uncomment and modify for your functions
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use forge::testing::{IsolatedTestDb, TestDatabase, TestMutationContext, TestQueryContext};
//     use std::path::Path;
//
//     async fn setup_db() -> IsolatedTestDb {
//         let base = TestDatabase::embedded().await.unwrap();
//         let db = base.isolated("my_test").await.unwrap();
//         db.migrate(Path::new("migrations")).await.unwrap();
//         db
//     }
//
//     #[tokio::test]
//     async fn test_my_query() {
//         let db = setup_db().await;
//         let ctx = TestQueryContext::builder()
//             .with_pool(db.pool().clone())
//             .as_user(Uuid::new_v4())
//             .build();
//
//         // Test your query here
//         db.cleanup().await.unwrap();
//     }
//
//     #[tokio::test]
//     async fn test_my_mutation() {
//         let db = setup_db().await;
//         let ctx = TestMutationContext::builder()
//             .with_pool(db.pool().clone())
//             .build();
//
//         // Test your mutation here
//         db.cleanup().await.unwrap();
//     }
// }
