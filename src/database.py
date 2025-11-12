import sqlite3
import pandas as pd
from datetime import datetime
import hashlib
import os
import json

class DatabaseManager:
    def __init__(self, db_path='data/gbs_system.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        os.makedirs('data', exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT DEFAULT 'clinician',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Patient records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                age INTEGER,
                gender TEXT,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                user_id INTEGER,
                input_data TEXT NOT NULL,
                predicted_subtype TEXT NOT NULL,
                confidence REAL NOT NULL,
                all_probabilities TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Sessions table for authentication
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, full_name, role='clinician'):
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, role))
            
            conn.commit()
            user_id = cursor.lastrowid
            print(f"✅ User {username} created successfully")
            return user_id
        except sqlite3.IntegrityError:
            print("❌ Username or email already exists")
            return None
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user and return user data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
            SELECT id, username, email, full_name, role FROM users 
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'role': user[4]
            }
        return None
    
    def create_session(self, user_id):
        """Create a new user session"""
        import secrets
        from datetime import datetime, timedelta
        
        session_token = secrets.token_hex(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user_id, session_token, expires_at))
        
        conn.commit()
        conn.close()
        
        return session_token
    
    def validate_session(self, session_token):
        """Validate session token and return user data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.username, u.email, u.full_name, u.role 
            FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.session_token = ? AND s.expires_at > CURRENT_TIMESTAMP AND u.is_active = 1
        ''', (session_token,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'role': user[4]
            }
        return None
    
    def save_prediction(self, user_id, input_data, prediction_result, patient_id=None):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO predictions (user_id, patient_id, input_data, predicted_subtype, confidence, all_probabilities)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                patient_id,
                json.dumps(input_data),
                prediction_result['predicted_subtype'],
                prediction_result['confidence'],
                json.dumps(prediction_result['all_probabilities'])
            ))
            
            conn.commit()
            prediction_id = cursor.lastrowid
            return prediction_id
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_predictions(self, user_id, limit=50):
        """Get prediction history for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, patient_id, input_data, predicted_subtype, confidence, created_at
            FROM predictions 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'id': row[0],
                'patient_id': row[1],
                'input_data': json.loads(row[2]),
                'predicted_subtype': row[3],
                'confidence': row[4],
                'created_at': row[5]
            })
        
        conn.close()
        return predictions

# Initialize database
db = DatabaseManager()