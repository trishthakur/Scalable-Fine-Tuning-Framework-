"""
Flask API for model predictions
Local deployment version
"""
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import pickle
import os
import threading
from config import API_HOST, API_PORT

def activate_api(model_path, host=API_HOST, port=API_PORT):
    """
    Activate the Flask API for model predictions
    
    Args:
        model_path (str): Path to the model file (.pkl)
        host (str): Host for the Flask app (default: 127.0.0.1)
        port (int): Port for the Flask app (default: 5000)
    
    Returns:
        str: URL of the running API
    """
    app = Flask(__name__)

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Prediction endpoint
        Expects a POST request with an image file
        """
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            # Preprocess the image
            img = Image.open(file)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((224, 224))  # Resize to MobileNet input size
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make predictions
            predictions = model.predict(img_array, verbose=0)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            confidence = predictions[0].tolist()

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "max_confidence": float(max(confidence))
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        """
        Health check endpoint
        """
        return jsonify({"status": "healthy", "model": model_path})

    # Start the Flask app in a separate thread
    def run_app():
        print(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)

    api_thread = threading.Thread(target=run_app, daemon=True)
    api_thread.start()

    # Return the API URL
    api_url = f"http://{host}:{port}/predict"
    print(f"API is running at: {api_url}")
    return api_url


if __name__ == '__main__':
    """
    Standalone mode - run the API directly
    Usage: python model_api.py
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_api.py <model_path>")
        print("Example: python model_api.py ./models/my_model.pkl")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Create a standalone Flask app
    app = Flask(__name__)
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            img = Image.open(file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            predicted_class = int(np.argmax(predictions, axis=1)[0])
            confidence = predictions[0].tolist()

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "max_confidence": float(max(confidence))
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "healthy", "model": model_path})
    
    print(f"Starting API server on {API_HOST}:{API_PORT}")
    print(f"API URL: http://{API_HOST}:{API_PORT}/predict")
    print("Press CTRL+C to stop")
    
    app.run(host=API_HOST, port=API_PORT, debug=False)