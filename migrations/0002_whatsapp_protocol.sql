-- @up

-- Device identity and crypto material
CREATE TABLE IF NOT EXISTS device (
    id SERIAL PRIMARY KEY,
    lid TEXT,
    pn TEXT,
    registration_id INTEGER,
    noise_key BYTEA,
    identity_key BYTEA,
    signed_pre_key BYTEA,
    signed_pre_key_id INTEGER,
    signed_pre_key_signature BYTEA,
    adv_secret_key BYTEA,
    account BYTEA,
    push_name TEXT,
    app_version_major INTEGER,
    app_version_minor INTEGER,
    app_version_patch INTEGER,
    edge_routing_info BYTEA,
    props_hash BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Signal protocol - Identity keys
CREATE TABLE IF NOT EXISTS identities (
    address TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    key BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (address, device_id)
);

-- Signal protocol - Sessions
CREATE TABLE IF NOT EXISTS sessions (
    address TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    record BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (address, device_id)
);

-- Signal protocol - Pre-keys
CREATE TABLE IF NOT EXISTS prekeys (
    id INTEGER NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    key BYTEA NOT NULL,
    uploaded BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, device_id)
);

-- Signal protocol - Signed pre-keys
CREATE TABLE IF NOT EXISTS signed_prekeys (
    id INTEGER NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    record BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, device_id)
);

-- Signal protocol - Sender keys (group messaging)
CREATE TABLE IF NOT EXISTS sender_keys (
    address TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    record BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (address, device_id)
);

-- App state synchronization - Sync keys
CREATE TABLE IF NOT EXISTS app_state_keys (
    key_id BYTEA NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    key_data BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (key_id, device_id)
);

-- App state synchronization - State versions
CREATE TABLE IF NOT EXISTS app_state_versions (
    name TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    state_data BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (name, device_id)
);

-- App state synchronization - Mutation MACs
CREATE TABLE IF NOT EXISTS app_state_mutation_macs (
    name TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    version BIGINT NOT NULL,
    index_mac BYTEA NOT NULL,
    value_mac BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (name, index_mac, device_id)
);

-- Protocol bookkeeping - SKDM recipients
CREATE TABLE IF NOT EXISTS skdm_recipients (
    group_jid TEXT NOT NULL,
    device_jid TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_jid, device_jid, device_id)
);

-- Protocol bookkeeping - LID-PN mapping
CREATE TABLE IF NOT EXISTS lid_pn_mapping (
    lid TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    learning_source TEXT,
    PRIMARY KEY (lid, device_id)
);

CREATE INDEX IF NOT EXISTS idx_lid_pn_mapping_phone_number ON lid_pn_mapping(phone_number, device_id);

-- Protocol bookkeeping - Base keys
CREATE TABLE IF NOT EXISTS base_keys (
    address TEXT NOT NULL,
    message_id TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    base_key BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (address, message_id, device_id)
);

-- Protocol bookkeeping - Device registry
CREATE TABLE IF NOT EXISTS device_registry (
    user_id TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    devices_json TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    phash TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, device_id)
);

-- Protocol bookkeeping - Sender key status
CREATE TABLE IF NOT EXISTS sender_key_status (
    group_jid TEXT NOT NULL,
    participant TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    marked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_jid, participant, device_id)
);

-- Protocol bookkeeping - TC tokens
CREATE TABLE IF NOT EXISTS tc_tokens (
    jid TEXT NOT NULL,
    device_id INTEGER NOT NULL REFERENCES device(id) ON DELETE CASCADE,
    token BYTEA NOT NULL,
    token_timestamp BIGINT NOT NULL,
    sender_timestamp BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (jid, device_id)
);

CREATE INDEX IF NOT EXISTS idx_tc_tokens_timestamp ON tc_tokens(token_timestamp, device_id);

-- Application-level user authentication
CREATE TABLE IF NOT EXISTS users (
    jid TEXT PRIMARY KEY,
    pairing_code TEXT,
    is_authenticated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at BIGINT NOT NULL,
    last_seen BIGINT NOT NULL
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_identities_device_id ON identities(device_id);
CREATE INDEX IF NOT EXISTS idx_sessions_device_id ON sessions(device_id);
CREATE INDEX IF NOT EXISTS idx_prekeys_device_id ON prekeys(device_id);
CREATE INDEX IF NOT EXISTS idx_signed_prekeys_device_id ON signed_prekeys(device_id);
CREATE INDEX IF NOT EXISTS idx_sender_keys_device_id ON sender_keys(device_id);
CREATE INDEX IF NOT EXISTS idx_app_state_keys_device_id ON app_state_keys(device_id);
CREATE INDEX IF NOT EXISTS idx_app_state_versions_device_id ON app_state_versions(device_id);
CREATE INDEX IF NOT EXISTS idx_base_keys_device_id ON base_keys(device_id);

-- Trigger to auto-update device updated_at
CREATE OR REPLACE FUNCTION update_device_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER device_updated_at
BEFORE UPDATE ON device
FOR EACH ROW
EXECUTE FUNCTION update_device_updated_at();

-- @down

DROP TRIGGER IF EXISTS device_updated_at ON device;
DROP FUNCTION IF EXISTS update_device_updated_at();

DROP TABLE IF EXISTS tc_tokens;
DROP TABLE IF EXISTS sender_key_status;
DROP TABLE IF EXISTS device_registry;
DROP TABLE IF EXISTS base_keys;
DROP TABLE IF EXISTS lid_pn_mapping;
DROP TABLE IF EXISTS skdm_recipients;
DROP TABLE IF EXISTS app_state_mutation_macs;
DROP TABLE IF EXISTS app_state_versions;
DROP TABLE IF EXISTS app_state_keys;
DROP TABLE IF EXISTS sender_keys;
DROP TABLE IF EXISTS signed_prekeys;
DROP TABLE IF EXISTS prekeys;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS identities;
DROP TABLE IF EXISTS device;
DROP TABLE IF EXISTS users;
