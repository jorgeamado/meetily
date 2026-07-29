-- User-created folders for organizing meetings in the sidebar (flat, one level)
CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Meetings may belong to at most one folder; NULL = top level
ALTER TABLE meetings ADD COLUMN folder_id TEXT;
