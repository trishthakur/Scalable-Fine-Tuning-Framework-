"""
Database operations for storing model metrics
Adapted for local SQLite deployment
"""
from config import get_connection

def save_model_metrics(model_version, accuracy, precision1, recall, f1):
    """
    Save model evaluation metrics to the database
    
    Args:
        model_version (str): Name/version of the model
        accuracy (float): Model accuracy
        precision1 (float): Model precision
        recall (float): Model recall
        f1 (float): Model F1 score
    """
    print(f"Saving metrics for {model_version}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision1:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            INSERT INTO model_metrics (model_version, accuracy, precision1, recall, f1) 
            VALUES (?, ?, ?, ?, ?)
        """, (model_version, accuracy, precision1, recall, f1))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("Model metrics saved successfully.")
    except Exception as e:
        print(f"Error saving model metrics: {e}")
        raise

def get_all_metrics():
    """
    Retrieve all model metrics from the database
    
    Returns:
        list: List of tuples containing model metrics
    """
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT id, model_version, accuracy, precision1, recall, f1, timestamp
            FROM model_metrics
            ORDER BY timestamp DESC
        """)
        
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return results
    except Exception as e:
        print(f"Error retrieving metrics: {e}")
        return []

def get_model_metrics(model_version):
    """
    Retrieve metrics for a specific model
    
    Args:
        model_version (str): Name/version of the model
        
    Returns:
        list: List of tuples containing model metrics
    """
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT id, model_version, accuracy, precision1, recall, f1, timestamp
            FROM model_metrics
            WHERE model_version = ?
            ORDER BY timestamp DESC
        """, (model_version,))
        
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return results
    except Exception as e:
        print(f"Error retrieving metrics for {model_version}: {e}")
        return []