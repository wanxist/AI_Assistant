-- Storage optimization: summary column, summarized flag, pagination index
ALTER TABLE t_session_info ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE t_session_message ADD COLUMN IF NOT EXISTS summarized BOOLEAN DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_session_message_sid_id ON t_session_message(session_id, id);
