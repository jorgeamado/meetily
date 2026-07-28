-- Add default speaker diarization settings to transcript_settings table
ALTER TABLE transcript_settings ADD COLUMN diarizationEnabled INTEGER NOT NULL DEFAULT 1;
ALTER TABLE transcript_settings ADD COLUMN diarizationNumSpeakers INTEGER;
ALTER TABLE transcript_settings ADD COLUMN diarizationSensitivity TEXT NOT NULL DEFAULT 'balanced';
