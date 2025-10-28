"""
Local storage utilities (replacement for Google Cloud Storage)
"""
import os
import shutil
from config import LOCAL_STORAGE_DIR

def save_model_locally(file_path, model_name):
    """
    Save a model file to local storage
    
    Args:
        file_path (str): Source file path
        model_name (str): Name for the saved model
        
    Returns:
        str: Path where the model was saved
    """
    try:
        destination = os.path.join(LOCAL_STORAGE_DIR, f"{model_name}.pkl")
        
        # Copy the file
        shutil.copy2(file_path, destination)
        
        print(f"Model saved locally at: {destination}")
        return destination
    except Exception as e:
        print(f"Error saving model locally: {e}")
        raise

def load_model_from_local(model_name):
    """
    Load a model file from local storage
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        str: Path to the model file
    """
    model_path = os.path.join(LOCAL_STORAGE_DIR, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found in local storage")
    
    return model_path

def list_saved_models():
    """
    List all saved models in local storage
    
    Returns:
        list: List of model names (without .pkl extension)
    """
    try:
        models = [f.replace('.pkl', '') for f in os.listdir(LOCAL_STORAGE_DIR) 
                  if f.endswith('.pkl')]
        return models
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def delete_model(model_name):
    """
    Delete a model from local storage
    
    Args:
        model_name (str): Name of the model to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        model_path = os.path.join(LOCAL_STORAGE_DIR, f"{model_name}.pkl")
        
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Model {model_name} deleted successfully")
            return True
        else:
            print(f"Model {model_name} not found")
            return False
    except Exception as e:
        print(f"Error deleting model: {e}")
        return False