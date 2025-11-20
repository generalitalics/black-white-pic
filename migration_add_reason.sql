-- Migration: Add reason field to user_progress table
-- Run this after init_db.sql if the table already exists

ALTER TABLE user_progress 
ADD COLUMN IF NOT EXISTS reason VARCHAR(20);

-- Update existing records to have a default reason based on status
UPDATE user_progress 
SET reason = CASE 
    WHEN status = 'completed' THEN 'completed'
    WHEN status = 'in_progress' THEN 'manual'
    ELSE 'manual'
END
WHERE reason IS NULL;

COMMENT ON COLUMN user_progress.reason IS 'Reason for progress update: manual, back, next_level, completed';

