-- @up
ALTER TABLE tc_tokens ALTER COLUMN sender_timestamp DROP NOT NULL;

-- @down
UPDATE tc_tokens SET sender_timestamp = 0 WHERE sender_timestamp IS NULL;
ALTER TABLE tc_tokens ALTER COLUMN sender_timestamp SET NOT NULL;
