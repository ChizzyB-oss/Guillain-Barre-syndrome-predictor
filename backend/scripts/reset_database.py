import os
import sys

# Add the backend/src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app import create_app
from src.models.database import db

def reset_database():
    """Reset the database completely"""
    print("🗑️  Resetting database...")
    
    app = create_app()
    
    with app.app_context():
        # Drop all tables and recreate
        db.drop_all()
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Create default users
        from src.models.database import User
        
        # Create admin user
        admin_user = User(
            username='admin',
            email='admin@gbssystem.com',
            full_name='System Administrator',
            role='admin'
        )
        admin_user.set_password('admin123')
        db.session.add(admin_user)
        
        # Create clinician user
        clinician_user = User(
            username='doctor1',
            email='doctor1@hospital.com',
            full_name='Dr. Sarah Johnson',
            role='clinician'
        )
        clinician_user.set_password('password123')
        db.session.add(clinician_user)
        
        # Create researcher user
        researcher_user = User(
            username='researcher1',
            email='researcher1@university.edu',
            full_name='Prof. Michael Chen',
            role='researcher'
        )
        researcher_user.set_password('password123')
        db.session.add(researcher_user)
        
        db.session.commit()
        print("✅ Default users created:")
        print("   - admin / admin123 (Administrator)")
        print("   - doctor1 / password123 (Clinician)")
        print("   - researcher1 / password123 (Researcher)")

if __name__ == '__main__':
    reset_database()