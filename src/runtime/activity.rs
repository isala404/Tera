//! Who is currently using the app-server.
//!
//! Memory maintenance has the lowest priority in the system: it may only run
//! when nothing else is, and must get out of the way the moment real work
//! arrives. That needs one shared answer to "is anything
//! active", which conversation turns and scheduled runs both contribute to.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

#[derive(Clone, Default)]
pub struct ActivityTracker {
    active: Arc<AtomicUsize>,
    /// Notified whenever work starts, so a maintenance run can be woken and
    /// interrupted rather than discovering it on the next poll.
    started: Arc<Notify>,
}

impl ActivityTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register work in flight. The returned guard clears it when dropped, so an
    /// early return or an error cannot leave the system looking permanently busy.
    pub fn begin(&self) -> ActivityGuard {
        self.active.fetch_add(1, Ordering::SeqCst);
        self.started.notify_waiters();
        ActivityGuard {
            active: self.active.clone(),
        }
    }

    pub fn active_count(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }

    pub fn is_idle(&self) -> bool {
        self.active_count() == 0
    }

    /// Resolves the next time any work starts.
    pub async fn wait_for_work(&self) {
        self.started.notified().await
    }
}

pub struct ActivityGuard {
    active: Arc<AtomicUsize>,
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idle_until_work_begins() {
        let activity = ActivityTracker::new();
        assert!(activity.is_idle());

        let guard = activity.begin();
        assert!(!activity.is_idle());

        drop(guard);
        assert!(activity.is_idle());
    }

    /// A failed turn must not leave maintenance deferred forever.
    #[test]
    fn test_concurrent_work_is_counted() {
        let activity = ActivityTracker::new();
        let a = activity.begin();
        let b = activity.begin();
        assert_eq!(activity.active_count(), 2);

        drop(a);
        assert!(!activity.is_idle());
        drop(b);
        assert!(activity.is_idle());
    }

    #[tokio::test]
    async fn test_waiter_is_woken_when_work_starts() {
        let activity = ActivityTracker::new();
        let waiter = activity.clone();

        let handle = tokio::spawn(async move { waiter.wait_for_work().await });
        tokio::task::yield_now().await;

        let _guard = activity.begin();
        tokio::time::timeout(std::time::Duration::from_secs(1), handle)
            .await
            .expect("waiter should have been woken")
            .unwrap();
    }
}
