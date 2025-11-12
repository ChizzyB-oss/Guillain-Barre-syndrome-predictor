import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.app import create_app
from src.models.database import db

def reset_database():
    """Reset the database completely"""
    app = create_app()
    
    with app.app_context():
        print("🗑️  Resetting database...")
        db.drop_all()
        db.create_all()
        print("✅ Database reset complete!")
        
        # You can add default data here if needed

if __name__ == '__main__':
    reset_database()