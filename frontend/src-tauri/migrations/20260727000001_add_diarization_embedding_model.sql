-- Add speaker embedding model selection to transcript_settings table
ALTER TABLE transcript_settings ADD COLUMN diarizationEmbeddingModel TEXT NOT NULL DEFAULT 'campplus';
