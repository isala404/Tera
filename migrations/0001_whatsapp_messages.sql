-- @up

-- pgvector extension is REQUIRED for this application
CREATE EXTENSION IF NOT EXISTS vector;

-- Create enum types for message direction and status
CREATE TYPE message_direction AS ENUM ('in', 'out');
CREATE TYPE message_status AS ENUM ('pending_agent', 'sent_agent', 'pending_gateway', 'sent_gateway', 'failed_gateway');

-- Create the whatsapp_messages table with pgvector support
CREATE TABLE IF NOT EXISTS whatsapp_messages (
    id UUID PRIMARY KEY,
    chat_id TEXT NOT NULL,
    direction message_direction NOT NULL,
    agent_id UUID,
    status message_status NOT NULL DEFAULT 'pending_agent',
    content_text TEXT,
    media JSONB,
    embedding vector(768),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create vector index for embeddings (using ivfflat for better performance with 768-dimensional vectors)
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_embedding ON whatsapp_messages USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_chat_id ON whatsapp_messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_agent_id ON whatsapp_messages(agent_id);
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_direction ON whatsapp_messages(direction);
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_status ON whatsapp_messages(status);
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_created_at ON whatsapp_messages(created_at);

-- Enable full-text search on content_text
CREATE INDEX IF NOT EXISTS idx_whatsapp_messages_content_text ON whatsapp_messages USING GIN(to_tsvector('english', content_text));

-- Create trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_whatsapp_messages_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER whatsapp_messages_updated_at
BEFORE UPDATE ON whatsapp_messages
FOR EACH ROW
EXECUTE FUNCTION update_whatsapp_messages_updated_at();

-- Enable Forge reactivity for real-time subscriptions
SELECT forge_enable_reactivity('whatsapp_messages');

-- @down

SELECT forge_disable_reactivity('whatsapp_messages');
DROP TRIGGER IF EXISTS whatsapp_messages_updated_at ON whatsapp_messages;
DROP FUNCTION IF EXISTS update_whatsapp_messages_updated_at();
DROP INDEX IF EXISTS idx_whatsapp_messages_embedding;
DROP INDEX IF EXISTS idx_whatsapp_messages_content_text;
DROP INDEX IF EXISTS idx_whatsapp_messages_created_at;
DROP INDEX IF EXISTS idx_whatsapp_messages_status;
DROP INDEX IF EXISTS idx_whatsapp_messages_direction;
DROP INDEX IF EXISTS idx_whatsapp_messages_agent_id;
DROP INDEX IF EXISTS idx_whatsapp_messages_chat_id;
DROP TABLE IF EXISTS whatsapp_messages;
DROP TYPE IF EXISTS message_status;
DROP TYPE IF EXISTS message_direction;
DROP EXTENSION IF EXISTS vector;
