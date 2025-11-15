import sqlite3
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="gbs_app.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT,
                password_hash TEXT,
                full_name TEXT,
                role TEXT,
                created_at TEXT
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                input_data TEXT,
                predicted_subtype TEXT,
                confidence REAL,
                all_probabilities TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        self.conn.commit()

    def add_user(self, user_data):
        try:
            self.cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_data["username"],
                user_data["email"],
                user_data["password_hash"],
                user_data["full_name"],
                user_data["role"],
                datetime.utcnow().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print("Database error:", e)
            return False

    def get_user_by_username(self, username):
        self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return self.cursor.fetchone()

db = DatabaseManager()
