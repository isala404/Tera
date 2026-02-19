#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MessageKey {
    pub chat_jid: String,
    pub message_id: String,
}

impl MessageKey {
    pub fn new(chat_jid: String, message_id: String) -> Self {
        Self {
            chat_jid,
            message_id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProcessingResult {
    pub response_text: String,
}

pub enum TaskState {
    InProgress { cancel_token: CancellationToken },
    Completed { result: ProcessingResult },
    Cancelled,
}

#[derive(Clone)]
pub struct TaskManager {
    tasks: Arc<RwLock<HashMap<MessageKey, TaskState>>>,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn start(&self, key: MessageKey) -> CancellationToken {
        let token = CancellationToken::new();
        self.tasks.write().await.insert(
            key,
            TaskState::InProgress {
                cancel_token: token.clone(),
            },
        );
        token
    }

    pub async fn cancel(&self, key: &MessageKey) -> bool {
        let tasks = self.tasks.read().await;
        if let Some(TaskState::InProgress { cancel_token, .. }) = tasks.get(key) {
            cancel_token.cancel();
            drop(tasks);
            self.tasks
                .write()
                .await
                .insert(key.clone(), TaskState::Cancelled);
            true
        } else {
            false
        }
    }

    pub async fn complete(&self, key: MessageKey, result: ProcessingResult) {
        self.tasks
            .write()
            .await
            .insert(key, TaskState::Completed { result });
    }

    pub async fn get(&self, key: &MessageKey) -> Option<TaskStateSnapshot> {
        let tasks = self.tasks.read().await;
        tasks.get(key).map(|state| match state {
            TaskState::InProgress { .. } => TaskStateSnapshot::InProgress,
            TaskState::Completed { result } => TaskStateSnapshot::Completed {
                result: result.clone(),
            },
            TaskState::Cancelled => TaskStateSnapshot::Cancelled,
        })
    }

    pub async fn cancel_all(&self) {
        let tasks = self.tasks.read().await;
        for state in tasks.values() {
            if let TaskState::InProgress { cancel_token, .. } = state {
                cancel_token.cancel();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum TaskStateSnapshot {
    InProgress,
    Completed { result: ProcessingResult },
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::{MessageKey, ProcessingResult, TaskManager, TaskStateSnapshot};

    #[tokio::test]
    async fn test_task_lifecycle() {
        let manager = TaskManager::new();
        let key = MessageKey::new("chat".to_string(), "msg".to_string());

        let token = manager.start(key.clone()).await;
        assert!(!token.is_cancelled());
        assert!(matches!(
            manager.get(&key).await,
            Some(TaskStateSnapshot::InProgress)
        ));

        manager
            .complete(
                key.clone(),
                ProcessingResult {
                    response_text: "ok".to_string(),
                },
            )
            .await;
        assert!(matches!(
            manager.get(&key).await,
            Some(TaskStateSnapshot::Completed { .. })
        ));
    }

    #[tokio::test]
    async fn test_task_cancel() {
        let manager = TaskManager::new();
        let key = MessageKey::new("chat".to_string(), "msg".to_string());
        manager.start(key.clone()).await;
        assert!(manager.cancel(&key).await);
        assert!(matches!(
            manager.get(&key).await,
            Some(TaskStateSnapshot::Cancelled)
        ));
    }
}
