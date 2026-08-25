use crate::transport::Transport;
use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};
use tracing::{debug, info, warn};

/// WhatsApp clears the "typing…" state on its own after a few seconds, so a
/// turn that runs for a minute needs the state refreshed to stay visible.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);

/// Keeps the typing indicator alive for as long as it is held.
///
/// Stopping is tied to the guard's lifetime rather than an explicit call: a turn
/// can end by error or early return, and an indicator that never clears leaves
/// the user watching the assistant type forever.
pub struct TypingGuard {
    transport: Arc<dyn Transport>,
    recipient: String,
    heartbeat: JoinHandle<()>,
}

impl TypingGuard {
    pub fn start(transport: Arc<dyn Transport>, recipient: String) -> Self {
        let beat_transport = transport.clone();
        let beat_recipient = recipient.clone();

        let heartbeat = tokio::spawn(async move {
            let mut announced_success = false;
            let mut announced_failure = false;
            loop {
                match beat_transport
                    .set_typing_status(&beat_recipient, true)
                    .await
                {
                    Ok(()) if !announced_success => {
                        info!("WhatsApp typing indicator active for {beat_recipient}");
                        announced_success = true;
                    }
                    Ok(()) => {}
                    Err(error) if !announced_failure => {
                        warn!("Could not activate WhatsApp typing indicator for {beat_recipient}: {error:?}");
                        announced_failure = true;
                    }
                    Err(error) => {
                        debug!("Could not refresh WhatsApp typing indicator for {beat_recipient}: {error:?}");
                    }
                }
                sleep(HEARTBEAT_INTERVAL).await;
            }
        });

        Self {
            transport,
            recipient,
            heartbeat,
        }
    }
}

impl Drop for TypingGuard {
    fn drop(&mut self) {
        self.heartbeat.abort();

        // Clearing is fire-and-forget: Drop cannot await, and a stale indicator
        // is not worth blocking the turn's completion path over.
        let transport = self.transport.clone();
        let recipient = std::mem::take(&mut self.recipient);
        tokio::spawn(async move {
            if let Err(error) = transport.set_typing_status(&recipient, false).await {
                warn!("Could not clear WhatsApp typing indicator for {recipient}: {error:?}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    #[tokio::test]
    async fn test_guard_starts_typing_and_clears_on_drop() {
        let mock = Arc::new(MockTransport::new());
        let transport: Arc<dyn Transport> = mock.clone();

        {
            let _guard = TypingGuard::start(transport, "owner@s.whatsapp.net".into());
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert_eq!(
                mock.typing_states.lock().unwrap().first(),
                Some(&("owner@s.whatsapp.net".to_string(), true))
            );
        }

        // Drop clears asynchronously.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            mock.typing_states.lock().unwrap().last(),
            Some(&("owner@s.whatsapp.net".to_string(), false))
        );
    }
}
