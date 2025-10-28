"""
Configuration file for local deployment
"""
import sqlite3
import os

# Local storage configuration
LOCAL_STORAGE_DIR = "./models"
LOCAL_DATABASE_DIR = "./database"

# Ensure directories exist
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
os.makedirs(LOCAL_DATABASE_DIR, exist_ok=True)

# Database configuration (SQLite for local deployment)
DATABASE_PATH = os.path.join(LOCAL_DATABASE_DIR, "metrics.db")

# API configuration
API_HOST = "127.0.0.1"
API_PORT = 5000

# Model configuration
DEFAULT_IMAGE_SIZE = (224, 224)
MAX_UPLOAD_SIZE_MB = 200

def get_connection():
    """
    Create a connection to the SQLite database
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    connection = sqlite3.connect(DATABASE_PATH)
    return connection

def init_database():
    """
    Initialize the database with required tables
    """
    connection = get_connection()
    cursor = connection.cursor()
    
    # Create model_metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            accuracy REAL NOT NULL,
            precision1 REAL NOT NULL,
            recall REAL NOT NULL,
            f1 REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    connection.commit()
    cursor.close()
    connection.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_database()