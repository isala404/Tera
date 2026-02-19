use async_trait::async_trait;
use prost::Message;
use sqlx::{PgPool, Row};
use std::sync::Arc;
use wacore::appstate::hash::HashState;
use wacore::appstate::processor::AppStateMutationMAC;
use wacore::libsignal::protocol::{KeyPair, PrivateKey, PublicKey};
use wacore::store::Device as CoreDevice;
use wacore::store::device::DEVICE_PROPS;
use wacore::store::error::{Result as StoreResult, StoreError};
use wacore::store::traits::{
    AppStateSyncKey, AppSyncStore, DeviceInfo, DeviceListRecord, DeviceStore, LidPnMappingEntry,
    ProtocolStore, SignalStore, TcTokenEntry,
};
use wacore_binary::jid::Jid;
use waproto::whatsapp as wa;

/// PostgreSQL-backed storage for WhatsApp protocol state.
#[derive(Clone)]
pub struct PostgresStore {
    pool: Arc<PgPool>,
    device_id: i32,
}

impl PostgresStore {
    #[allow(dead_code)]
    pub fn new(pool: Arc<PgPool>) -> Self {
        Self { pool, device_id: 1 }
    }

    #[allow(dead_code)]
    pub fn new_for_device(pool: Arc<PgPool>, device_id: i32) -> Self {
        Self { pool, device_id }
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> Arc<PgPool> {
        Arc::clone(&self.pool)
    }

    fn serialize_keypair(key_pair: &KeyPair) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(key_pair.private_key.serialize());
        bytes.extend_from_slice(key_pair.public_key.public_key_bytes());
        bytes
    }

    fn deserialize_keypair(bytes: &[u8]) -> StoreResult<KeyPair> {
        if bytes.len() != 64 {
            return Err(StoreError::Serialization(format!(
                "Invalid keypair length: {}",
                bytes.len()
            )));
        }

        let private_key = PrivateKey::deserialize(&bytes[0..32])
            .map_err(|e| StoreError::Serialization(e.to_string()))?;
        let public_key = PublicKey::from_djb_public_key_bytes(&bytes[32..64])
            .map_err(|e| StoreError::Serialization(e.to_string()))?;
        Ok(KeyPair::new(public_key, private_key))
    }

    async fn save_device_data_for_device(
        &self,
        device_id: i32,
        device_data: &CoreDevice,
    ) -> StoreResult<()> {
        let noise_key = Self::serialize_keypair(&device_data.noise_key);
        let identity_key = Self::serialize_keypair(&device_data.identity_key);
        let signed_pre_key = Self::serialize_keypair(&device_data.signed_pre_key);
        let account = device_data.account.as_ref().map(|a| a.encode_to_vec());
        let lid = device_data
            .lid
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let pn = device_data
            .pn
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let props_hash = device_data
            .props_hash
            .as_ref()
            .map(|s| s.as_bytes().to_vec());

        sqlx::query(
            r#"
            INSERT INTO device (
                id, lid, pn, registration_id, noise_key, identity_key, signed_pre_key,
                signed_pre_key_id, signed_pre_key_signature, adv_secret_key, account, push_name,
                app_version_major, app_version_minor, app_version_patch, edge_routing_info, props_hash
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (id) DO UPDATE SET
                lid = EXCLUDED.lid,
                pn = EXCLUDED.pn,
                registration_id = EXCLUDED.registration_id,
                noise_key = EXCLUDED.noise_key,
                identity_key = EXCLUDED.identity_key,
                signed_pre_key = EXCLUDED.signed_pre_key,
                signed_pre_key_id = EXCLUDED.signed_pre_key_id,
                signed_pre_key_signature = EXCLUDED.signed_pre_key_signature,
                adv_secret_key = EXCLUDED.adv_secret_key,
                account = EXCLUDED.account,
                push_name = EXCLUDED.push_name,
                app_version_major = EXCLUDED.app_version_major,
                app_version_minor = EXCLUDED.app_version_minor,
                app_version_patch = EXCLUDED.app_version_patch,
                edge_routing_info = EXCLUDED.edge_routing_info,
                props_hash = EXCLUDED.props_hash,
                updated_at = NOW()
            "#,
        )
        .bind(device_id)
        .bind(lid)
        .bind(pn)
        .bind(device_data.registration_id as i32)
        .bind(noise_key)
        .bind(identity_key)
        .bind(signed_pre_key)
        .bind(device_data.signed_pre_key_id as i32)
        .bind(device_data.signed_pre_key_signature.to_vec())
        .bind(device_data.adv_secret_key.to_vec())
        .bind(account)
        .bind(device_data.push_name.clone())
        .bind(device_data.app_version_primary as i32)
        .bind(device_data.app_version_secondary as i32)
        .bind(device_data.app_version_tertiary as i32)
        .bind(device_data.edge_routing_info.clone())
        .bind(props_hash)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    async fn load_device_data_for_device(&self, device_id: i32) -> StoreResult<Option<CoreDevice>> {
        let row = sqlx::query(
            r#"
            SELECT
                lid, pn, registration_id, noise_key, identity_key, signed_pre_key,
                signed_pre_key_id, signed_pre_key_signature, adv_secret_key, account, push_name,
                app_version_major, app_version_minor, app_version_patch, edge_routing_info, props_hash
            FROM device
            WHERE id = $1
            "#,
        )
        .bind(device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        let Some(row) = row else {
            return Ok(None);
        };

        let lid: String = row.get("lid");
        let pn: String = row.get("pn");
        let noise_key: Vec<u8> = row.get("noise_key");
        let identity_key: Vec<u8> = row.get("identity_key");
        let signed_pre_key: Vec<u8> = row.get("signed_pre_key");
        let signed_pre_key_signature: Vec<u8> = row.get("signed_pre_key_signature");
        let adv_secret_key: Vec<u8> = row.get("adv_secret_key");
        let account: Option<Vec<u8>> = row.get("account");
        let props_hash_bytes: Option<Vec<u8>> = row.get("props_hash");

        let signed_pre_key_signature: [u8; 64] =
            signed_pre_key_signature.try_into().map_err(|_| {
                StoreError::Serialization("Invalid signed_pre_key_signature length".to_string())
            })?;
        let adv_secret_key: [u8; 32] = adv_secret_key
            .try_into()
            .map_err(|_| StoreError::Serialization("Invalid adv_secret_key length".to_string()))?;

        let account = account
            .map(|value| {
                wa::AdvSignedDeviceIdentity::decode(&value[..])
                    .map_err(|e| StoreError::Serialization(e.to_string()))
            })
            .transpose()?;

        Ok(Some(CoreDevice {
            pn: if pn.is_empty() {
                None
            } else {
                pn.parse::<Jid>().ok()
            },
            lid: if lid.is_empty() {
                None
            } else {
                lid.parse::<Jid>().ok()
            },
            registration_id: row.get::<i32, _>("registration_id") as u32,
            noise_key: Self::deserialize_keypair(&noise_key)?,
            identity_key: Self::deserialize_keypair(&identity_key)?,
            signed_pre_key: Self::deserialize_keypair(&signed_pre_key)?,
            signed_pre_key_id: row.get::<i32, _>("signed_pre_key_id") as u32,
            signed_pre_key_signature,
            adv_secret_key,
            account,
            push_name: row.get("push_name"),
            app_version_primary: row.get::<i32, _>("app_version_major") as u32,
            app_version_secondary: row.get::<i32, _>("app_version_minor") as u32,
            app_version_tertiary: row.get::<i32, _>("app_version_patch") as u32,
            app_version_last_fetched_ms: 0,
            device_props: DEVICE_PROPS.clone(),
            edge_routing_info: row.get("edge_routing_info"),
            props_hash: props_hash_bytes.and_then(|v| String::from_utf8(v).ok()),
        }))
    }

    async fn device_exists(&self, device_id: i32) -> StoreResult<bool> {
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM device WHERE id = $1")
            .bind(device_id)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(count > 0)
    }

    async fn create_new_device(&self) -> StoreResult<i32> {
        let device = CoreDevice::new();
        self.save_device_data_for_device(self.device_id, &device)
            .await?;
        Ok(self.device_id)
    }
}

#[async_trait]
impl SignalStore for PostgresStore {
    async fn put_identity(&self, address: &str, key: [u8; 32]) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO identities (address, device_id, key)
            VALUES ($1, $2, $3)
            ON CONFLICT (address, device_id) DO UPDATE SET key = EXCLUDED.key, updated_at = NOW()
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .bind(key.to_vec())
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn load_identity(&self, address: &str) -> StoreResult<Option<Vec<u8>>> {
        sqlx::query_scalar(
            r#"
            SELECT key
            FROM identities
            WHERE address = $1 AND device_id = $2
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn delete_identity(&self, address: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM identities
            WHERE address = $1 AND device_id = $2
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_session(&self, address: &str) -> StoreResult<Option<Vec<u8>>> {
        sqlx::query_scalar(
            r#"
            SELECT record
            FROM sessions
            WHERE address = $1 AND device_id = $2
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn put_session(&self, address: &str, session: &[u8]) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO sessions (address, device_id, record)
            VALUES ($1, $2, $3)
            ON CONFLICT (address, device_id) DO UPDATE SET record = EXCLUDED.record, updated_at = NOW()
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .bind(session.to_vec())
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn delete_session(&self, address: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM sessions
            WHERE address = $1 AND device_id = $2
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn store_prekey(&self, id: u32, record: &[u8], uploaded: bool) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO prekeys (id, device_id, key, uploaded)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id, device_id) DO UPDATE SET
                key = EXCLUDED.key,
                uploaded = EXCLUDED.uploaded,
                updated_at = NOW()
            "#,
        )
        .bind(id as i32)
        .bind(self.device_id)
        .bind(record.to_vec())
        .bind(uploaded)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn load_prekey(&self, id: u32) -> StoreResult<Option<Vec<u8>>> {
        sqlx::query_scalar(
            r#"
            SELECT key
            FROM prekeys
            WHERE id = $1 AND device_id = $2
            "#,
        )
        .bind(id as i32)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn remove_prekey(&self, id: u32) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM prekeys
            WHERE id = $1 AND device_id = $2
            "#,
        )
        .bind(id as i32)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn store_signed_prekey(&self, id: u32, record: &[u8]) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO signed_prekeys (id, device_id, record)
            VALUES ($1, $2, $3)
            ON CONFLICT (id, device_id) DO UPDATE SET
                record = EXCLUDED.record,
                updated_at = NOW()
            "#,
        )
        .bind(id as i32)
        .bind(self.device_id)
        .bind(record.to_vec())
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn load_signed_prekey(&self, id: u32) -> StoreResult<Option<Vec<u8>>> {
        sqlx::query_scalar(
            r#"
            SELECT record
            FROM signed_prekeys
            WHERE id = $1 AND device_id = $2
            "#,
        )
        .bind(id as i32)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn load_all_signed_prekeys(&self) -> StoreResult<Vec<(u32, Vec<u8>)>> {
        let rows = sqlx::query_as::<_, (i32, Vec<u8>)>(
            r#"
            SELECT id, record
            FROM signed_prekeys
            WHERE device_id = $1
            ORDER BY id
            "#,
        )
        .bind(self.device_id)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|(id, record)| (id as u32, record))
            .collect())
    }

    async fn remove_signed_prekey(&self, id: u32) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM signed_prekeys
            WHERE id = $1 AND device_id = $2
            "#,
        )
        .bind(id as i32)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn put_sender_key(&self, address: &str, record: &[u8]) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO sender_keys (address, device_id, record)
            VALUES ($1, $2, $3)
            ON CONFLICT (address, device_id) DO UPDATE SET
                record = EXCLUDED.record,
                updated_at = NOW()
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .bind(record.to_vec())
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_sender_key(&self, address: &str) -> StoreResult<Option<Vec<u8>>> {
        sqlx::query_scalar(
            r#"
            SELECT record
            FROM sender_keys
            WHERE address = $1 AND device_id = $2
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn delete_sender_key(&self, address: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM sender_keys
            WHERE address = $1 AND device_id = $2
            "#,
        )
        .bind(address)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl AppSyncStore for PostgresStore {
    async fn get_sync_key(&self, key_id: &[u8]) -> StoreResult<Option<AppStateSyncKey>> {
        let stored: Option<Vec<u8>> = sqlx::query_scalar(
            r#"
            SELECT key_data
            FROM app_state_keys
            WHERE key_id = $1 AND device_id = $2
            "#,
        )
        .bind(key_id)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        stored
            .map(|value| {
                bincode::serde::decode_from_slice(&value, bincode::config::standard())
                    .map(|(decoded, _): (AppStateSyncKey, usize)| decoded)
                    .map_err(|e| StoreError::Serialization(e.to_string()))
            })
            .transpose()
    }

    async fn set_sync_key(&self, key_id: &[u8], key: AppStateSyncKey) -> StoreResult<()> {
        let encoded = bincode::serde::encode_to_vec(&key, bincode::config::standard())
            .map_err(|e| StoreError::Serialization(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO app_state_keys (key_id, device_id, key_data)
            VALUES ($1, $2, $3)
            ON CONFLICT (key_id, device_id) DO UPDATE SET
                key_data = EXCLUDED.key_data,
                updated_at = NOW()
            "#,
        )
        .bind(key_id)
        .bind(self.device_id)
        .bind(encoded)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    async fn get_version(&self, name: &str) -> StoreResult<HashState> {
        let state_data: Option<Vec<u8>> = sqlx::query_scalar(
            r#"
            SELECT state_data
            FROM app_state_versions
            WHERE name = $1 AND device_id = $2
            "#,
        )
        .bind(name)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        match state_data {
            Some(state_data) => {
                bincode::serde::decode_from_slice(&state_data, bincode::config::standard())
                    .map(|(decoded, _): (HashState, usize)| decoded)
                    .map_err(|e| StoreError::Serialization(e.to_string()))
            }
            None => Ok(HashState::default()),
        }
    }

    async fn set_version(&self, name: &str, state: HashState) -> StoreResult<()> {
        let encoded = bincode::serde::encode_to_vec(&state, bincode::config::standard())
            .map_err(|e| StoreError::Serialization(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO app_state_versions (name, device_id, state_data)
            VALUES ($1, $2, $3)
            ON CONFLICT (name, device_id) DO UPDATE SET
                state_data = EXCLUDED.state_data,
                updated_at = NOW()
            "#,
        )
        .bind(name)
        .bind(self.device_id)
        .bind(encoded)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn put_mutation_macs(
        &self,
        name: &str,
        version: u64,
        mutations: &[AppStateMutationMAC],
    ) -> StoreResult<()> {
        for mutation in mutations {
            sqlx::query(
                r#"
                INSERT INTO app_state_mutation_macs (name, device_id, version, index_mac, value_mac)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (name, index_mac, device_id) DO UPDATE SET
                    version = EXCLUDED.version,
                    value_mac = EXCLUDED.value_mac,
                    updated_at = NOW()
                "#,
            )
            .bind(name)
            .bind(self.device_id)
            .bind(version as i64)
            .bind(mutation.index_mac.clone())
            .bind(mutation.value_mac.clone())
            .execute(&*self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        Ok(())
    }

    async fn get_mutation_mac(&self, name: &str, index_mac: &[u8]) -> StoreResult<Option<Vec<u8>>> {
        sqlx::query_scalar(
            r#"
            SELECT value_mac
            FROM app_state_mutation_macs
            WHERE name = $1 AND index_mac = $2 AND device_id = $3
            "#,
        )
        .bind(name)
        .bind(index_mac)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn delete_mutation_macs(&self, name: &str, index_macs: &[Vec<u8>]) -> StoreResult<()> {
        for index_mac in index_macs {
            sqlx::query(
                r#"
                DELETE FROM app_state_mutation_macs
                WHERE name = $1 AND index_mac = $2 AND device_id = $3
                "#,
            )
            .bind(name)
            .bind(index_mac)
            .bind(self.device_id)
            .execute(&*self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        Ok(())
    }
}

#[async_trait]
impl ProtocolStore for PostgresStore {
    async fn get_skdm_recipients(&self, group_jid: &str) -> StoreResult<Vec<Jid>> {
        let values: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT device_jid
            FROM skdm_recipients
            WHERE group_jid = $1 AND device_id = $2
            "#,
        )
        .bind(group_jid)
        .bind(self.device_id)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(values
            .into_iter()
            .filter_map(|value| match value.parse::<Jid>() {
                Ok(jid) => Some(jid),
                Err(err) => {
                    tracing::warn!("Failed to parse SKDM recipient JID '{}': {}", value, err);
                    None
                }
            })
            .collect())
    }

    async fn add_skdm_recipients(&self, group_jid: &str, device_jids: &[Jid]) -> StoreResult<()> {
        for device_jid in device_jids {
            sqlx::query(
                r#"
                INSERT INTO skdm_recipients (group_jid, device_jid, device_id, created_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (group_jid, device_jid, device_id) DO NOTHING
                "#,
            )
            .bind(group_jid)
            .bind(device_jid.to_string())
            .bind(self.device_id)
            .execute(&*self.pool)
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;
        }

        Ok(())
    }

    async fn clear_skdm_recipients(&self, group_jid: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM skdm_recipients
            WHERE group_jid = $1 AND device_id = $2
            "#,
        )
        .bind(group_jid)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_lid_mapping(&self, lid: &str) -> StoreResult<Option<LidPnMappingEntry>> {
        let row = sqlx::query(
            r#"
            SELECT
                lid,
                phone_number,
                EXTRACT(EPOCH FROM created_at)::BIGINT AS created_at,
                learning_source,
                EXTRACT(EPOCH FROM updated_at)::BIGINT AS updated_at
            FROM lid_pn_mapping
            WHERE lid = $1 AND device_id = $2
            "#,
        )
        .bind(lid)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|row| LidPnMappingEntry {
            lid: row.get("lid"),
            phone_number: row.get("phone_number"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            learning_source: row.get("learning_source"),
        }))
    }

    async fn get_pn_mapping(&self, phone: &str) -> StoreResult<Option<LidPnMappingEntry>> {
        let row = sqlx::query(
            r#"
            SELECT
                lid,
                phone_number,
                EXTRACT(EPOCH FROM created_at)::BIGINT AS created_at,
                learning_source,
                EXTRACT(EPOCH FROM updated_at)::BIGINT AS updated_at
            FROM lid_pn_mapping
            WHERE phone_number = $1 AND device_id = $2
            ORDER BY updated_at DESC
            LIMIT 1
            "#,
        )
        .bind(phone)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|row| LidPnMappingEntry {
            lid: row.get("lid"),
            phone_number: row.get("phone_number"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            learning_source: row.get("learning_source"),
        }))
    }

    async fn put_lid_mapping(&self, entry: &LidPnMappingEntry) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO lid_pn_mapping (
                lid, phone_number, device_id, created_at, updated_at, learning_source
            )
            VALUES (
                $1, $2, $3, to_timestamp($4), to_timestamp($5), $6
            )
            ON CONFLICT (lid, device_id) DO UPDATE SET
                phone_number = EXCLUDED.phone_number,
                updated_at = EXCLUDED.updated_at,
                learning_source = EXCLUDED.learning_source
            "#,
        )
        .bind(&entry.lid)
        .bind(&entry.phone_number)
        .bind(self.device_id)
        .bind(entry.created_at as f64)
        .bind(entry.updated_at as f64)
        .bind(&entry.learning_source)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_all_lid_mappings(&self) -> StoreResult<Vec<LidPnMappingEntry>> {
        let rows = sqlx::query(
            r#"
            SELECT
                lid,
                phone_number,
                EXTRACT(EPOCH FROM created_at)::BIGINT AS created_at,
                learning_source,
                EXTRACT(EPOCH FROM updated_at)::BIGINT AS updated_at
            FROM lid_pn_mapping
            WHERE device_id = $1
            "#,
        )
        .bind(self.device_id)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(rows
            .into_iter()
            .map(|row| LidPnMappingEntry {
                lid: row.get("lid"),
                phone_number: row.get("phone_number"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                learning_source: row.get("learning_source"),
            })
            .collect())
    }

    async fn save_base_key(
        &self,
        address: &str,
        message_id: &str,
        base_key: &[u8],
    ) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO base_keys (address, message_id, device_id, base_key, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (address, message_id, device_id) DO UPDATE SET
                base_key = EXCLUDED.base_key
            "#,
        )
        .bind(address)
        .bind(message_id)
        .bind(self.device_id)
        .bind(base_key)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn has_same_base_key(
        &self,
        address: &str,
        message_id: &str,
        current_base_key: &[u8],
    ) -> StoreResult<bool> {
        let stored: Option<Vec<u8>> = sqlx::query_scalar(
            r#"
            SELECT base_key
            FROM base_keys
            WHERE address = $1 AND message_id = $2 AND device_id = $3
            "#,
        )
        .bind(address)
        .bind(message_id)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(stored.as_deref() == Some(current_base_key))
    }

    async fn delete_base_key(&self, address: &str, message_id: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM base_keys
            WHERE address = $1 AND message_id = $2 AND device_id = $3
            "#,
        )
        .bind(address)
        .bind(message_id)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn update_device_list(&self, record: DeviceListRecord) -> StoreResult<()> {
        let devices_json = serde_json::to_string(&record.devices)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;

        sqlx::query(
            r#"
            INSERT INTO device_registry (user_id, device_id, devices_json, timestamp, phash, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (user_id, device_id) DO UPDATE SET
                devices_json = EXCLUDED.devices_json,
                timestamp = EXCLUDED.timestamp,
                phash = EXCLUDED.phash,
                updated_at = NOW()
            "#,
        )
        .bind(record.user)
        .bind(self.device_id)
        .bind(devices_json)
        .bind(record.timestamp)
        .bind(record.phash)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    async fn get_devices(&self, user: &str) -> StoreResult<Option<DeviceListRecord>> {
        let row = sqlx::query(
            r#"
            SELECT user_id, devices_json, timestamp, phash
            FROM device_registry
            WHERE user_id = $1 AND device_id = $2
            "#,
        )
        .bind(user)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        let Some(row) = row else {
            return Ok(None);
        };

        let devices_json: String = row.get("devices_json");
        let devices: Vec<DeviceInfo> = serde_json::from_str(&devices_json)
            .map_err(|e| StoreError::Serialization(e.to_string()))?;

        Ok(Some(DeviceListRecord {
            user: row.get("user_id"),
            devices,
            timestamp: row.get("timestamp"),
            phash: row.get("phash"),
        }))
    }

    async fn mark_forget_sender_key(&self, group_jid: &str, participant: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO sender_key_status (group_jid, participant, device_id, marked_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (group_jid, participant, device_id) DO UPDATE SET
                marked_at = NOW()
            "#,
        )
        .bind(group_jid)
        .bind(participant)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn consume_forget_marks(&self, group_jid: &str) -> StoreResult<Vec<String>> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        let participants: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT participant
            FROM sender_key_status
            WHERE group_jid = $1 AND device_id = $2
            "#,
        )
        .bind(group_jid)
        .bind(self.device_id)
        .fetch_all(&mut *tx)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        sqlx::query(
            r#"
            DELETE FROM sender_key_status
            WHERE group_jid = $1 AND device_id = $2
            "#,
        )
        .bind(group_jid)
        .bind(self.device_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(participants)
    }

    async fn get_tc_token(&self, jid: &str) -> StoreResult<Option<TcTokenEntry>> {
        let row = sqlx::query(
            r#"
            SELECT token, token_timestamp, sender_timestamp
            FROM tc_tokens
            WHERE jid = $1 AND device_id = $2
            "#,
        )
        .bind(jid)
        .bind(self.device_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(row.map(|row| TcTokenEntry {
            token: row.get("token"),
            token_timestamp: row.get("token_timestamp"),
            sender_timestamp: row.get("sender_timestamp"),
        }))
    }

    async fn put_tc_token(&self, jid: &str, entry: &TcTokenEntry) -> StoreResult<()> {
        sqlx::query(
            r#"
            INSERT INTO tc_tokens (jid, device_id, token, token_timestamp, sender_timestamp, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (jid, device_id) DO UPDATE SET
                token = EXCLUDED.token,
                token_timestamp = EXCLUDED.token_timestamp,
                sender_timestamp = EXCLUDED.sender_timestamp,
                updated_at = NOW()
            "#,
        )
        .bind(jid)
        .bind(self.device_id)
        .bind(entry.token.clone())
        .bind(entry.token_timestamp)
        .bind(entry.sender_timestamp)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(())
    }

    async fn delete_tc_token(&self, jid: &str) -> StoreResult<()> {
        sqlx::query(
            r#"
            DELETE FROM tc_tokens
            WHERE jid = $1 AND device_id = $2
            "#,
        )
        .bind(jid)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;
        Ok(())
    }

    async fn get_all_tc_token_jids(&self) -> StoreResult<Vec<String>> {
        sqlx::query_scalar(
            r#"
            SELECT jid
            FROM tc_tokens
            WHERE device_id = $1
            "#,
        )
        .bind(self.device_id)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))
    }

    async fn delete_expired_tc_tokens(&self, cutoff_timestamp: i64) -> StoreResult<u32> {
        let result = sqlx::query(
            r#"
            DELETE FROM tc_tokens
            WHERE token_timestamp < $1 AND device_id = $2
            "#,
        )
        .bind(cutoff_timestamp)
        .bind(self.device_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StoreError::Database(e.to_string()))?;

        Ok(result.rows_affected() as u32)
    }
}

#[async_trait]
impl DeviceStore for PostgresStore {
    async fn save(&self, device: &CoreDevice) -> StoreResult<()> {
        self.save_device_data_for_device(self.device_id, device)
            .await
    }

    async fn load(&self) -> StoreResult<Option<CoreDevice>> {
        self.load_device_data_for_device(self.device_id).await
    }

    async fn exists(&self) -> StoreResult<bool> {
        self.device_exists(self.device_id).await
    }

    async fn create(&self) -> StoreResult<i32> {
        self.create_new_device().await
    }
}

#[cfg(test)]
mod tests {
    use super::PostgresStore;
    use crate::utils::test_db::setup::{init_test_db_with_vector, is_pgvector_unavailable};
    use forge::testing::IsolatedTestDb;
    use std::sync::Arc;
    use wacore::store::traits::{DeviceStore, LidPnMappingEntry, ProtocolStore, SignalStore};

    struct TestStore {
        _db: IsolatedTestDb,
        store: PostgresStore,
    }

    async fn setup_store(name: &str) -> Option<TestStore> {
        match init_test_db_with_vector(name).await {
            Ok(db) => {
                let store = PostgresStore::new(Arc::new(db.pool().clone()));
                Some(TestStore { _db: db, store })
            }
            Err(err) if is_pgvector_unavailable(err.as_ref()) => {
                eprintln!("Skipping whatsapp_store DB test: {}", err);
                None
            }
            Err(err) => panic!("test db should initialize: {}", err),
        }
    }

    #[tokio::test]
    async fn test_device_create_and_load() {
        let Some(test_store) = setup_store("wa_store_device").await else {
            return;
        };
        let store = test_store.store;
        assert!(!store.exists().await.expect("exists should work"));

        let id = store.create().await.expect("create should work");
        assert_eq!(id, 1);
        assert!(store.exists().await.expect("exists should work"));

        let loaded = store
            .load()
            .await
            .expect("load should work")
            .expect("device should exist");
        assert!(loaded.registration_id > 0);
    }

    #[tokio::test]
    async fn test_identity_roundtrip() {
        let Some(test_store) = setup_store("wa_store_identity").await else {
            return;
        };
        let store = test_store.store;
        store.create().await.expect("create should work");

        let key = [42u8; 32];
        store
            .put_identity("user@s.whatsapp.net", key)
            .await
            .expect("put identity should work");

        let loaded = store
            .load_identity("user@s.whatsapp.net")
            .await
            .expect("load identity should work");
        assert_eq!(loaded, Some(key.to_vec()));

        store
            .delete_identity("user@s.whatsapp.net")
            .await
            .expect("delete identity should work");
        assert!(
            store
                .load_identity("user@s.whatsapp.net")
                .await
                .expect("load identity should work")
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_lid_mapping_roundtrip() {
        let Some(test_store) = setup_store("wa_store_lid_mapping").await else {
            return;
        };
        let store = test_store.store;
        store.create().await.expect("create should work");

        let now = 1_700_000_000_i64;
        let entry = LidPnMappingEntry {
            lid: "100000012345678".to_string(),
            phone_number: "15551234567".to_string(),
            created_at: now,
            updated_at: now,
            learning_source: "test".to_string(),
        };

        store
            .put_lid_mapping(&entry)
            .await
            .expect("put_lid_mapping should work");

        let by_lid = store
            .get_lid_mapping(&entry.lid)
            .await
            .expect("get_lid_mapping should work")
            .expect("mapping should exist");
        assert_eq!(by_lid.phone_number, entry.phone_number);

        let by_pn = store
            .get_pn_mapping(&entry.phone_number)
            .await
            .expect("get_pn_mapping should work")
            .expect("mapping should exist");
        assert_eq!(by_pn.lid, entry.lid);
    }
}
