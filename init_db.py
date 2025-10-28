"""
Database initialization script for local deployment
Run this script before starting the application
"""
from config import init_database

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    print("Database initialization complete!")
    print("You can now run the application with: streamlit run FineTuneMobileNet.py")