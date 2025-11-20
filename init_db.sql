-- PostgreSQL database initialization script for nonogram game

-- Drop tables if they exist (in reverse order of dependencies)
DROP TABLE IF EXISTS user_progress CASCADE;
DROP TABLE IF EXISTS levels CASCADE;
DROP TABLE IF EXISTS difficulty CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Create users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create difficulty table
CREATE TABLE difficulty (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20) UNIQUE NOT NULL
);

-- Create levels table
CREATE TABLE levels (
    id SERIAL PRIMARY KEY,
    difficulty_id INTEGER NOT NULL REFERENCES difficulty(id) ON DELETE CASCADE,
    number INTEGER NOT NULL,
    UNIQUE(difficulty_id, number)
);

-- Create user_progress table
CREATE TABLE user_progress (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    level_id INTEGER NOT NULL REFERENCES levels(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'not_started', -- not_started, in_progress, completed
    reason VARCHAR(20), -- manual, back, next_level, completed
    matrix JSONB, -- stores the solution matrix as JSON
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, level_id)
);

-- Create indexes for better query performance
CREATE INDEX idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX idx_user_progress_level_id ON user_progress(level_id);
CREATE INDEX idx_user_progress_status ON user_progress(status);
CREATE INDEX idx_levels_difficulty_id ON levels(difficulty_id);

-- Insert initial data

-- Insert difficulty types
INSERT INTO difficulty (type) VALUES ('easy'), ('medium'), ('hard');

-- Insert example users
-- Note: In production, passwords should be hashed (e.g., using bcrypt)
-- These are example plaintext passwords - replace with hashed versions in production!
INSERT INTO users (username, password, is_admin) VALUES
    ('player1', 'password123', TRUE),
    ('player2', 'secret456', FALSE);

-- Insert some example levels
-- Easy levels
INSERT INTO levels (difficulty_id, number) 
SELECT d.id, num FROM difficulty d, generate_series(1, 5) AS num WHERE d.type = 'easy';

-- Medium levels
INSERT INTO levels (difficulty_id, number) 
SELECT d.id, num FROM difficulty d, generate_series(1, 10) AS num WHERE d.type = 'medium';

-- Hard levels
INSERT INTO levels (difficulty_id, number) 
SELECT d.id, num FROM difficulty d, generate_series(1, 15) AS num WHERE d.type = 'hard';

-- Insert example user progress (optional - can be empty initially)
-- INSERT INTO user_progress (user_id, level_id, status) 
-- SELECT u.id, l.id, 'not_started' 
-- FROM users u, levels l 
-- WHERE l.difficulty_id = (SELECT id FROM difficulty WHERE type = 'easy') AND l.number = 1;

COMMENT ON TABLE users IS 'User accounts for the nonogram game';
COMMENT ON TABLE difficulty IS 'Difficulty levels: easy, medium, hard';
COMMENT ON TABLE levels IS 'Game levels organized by difficulty';
COMMENT ON TABLE user_progress IS 'Tracks user progress through levels, stores solution matrices';


