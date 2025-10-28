"""
API client for making predictions
Example usage script
"""
import requests
import json
import sys
import os

def predict_image(image_path, api_url="http://127.0.0.1:5000/predict"):
    """
    Send an image to the API and get predictions
    
    Args:
        image_path (str): Path to the image file
        api_url (str): URL of the prediction API
        
    Returns:
        dict: Prediction results
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    print(f"Sending image to API: {api_url}")
    print(f"Image: {image_path}")
    
    try:
        # Send the image file to the server
        with open(image_path, 'rb') as img:
            files = {'file': img}
            response = requests.post(api_url, files=files)

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful!")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Max Confidence: {result['max_confidence']:.2%}")
            print(f"\nAll Confidence Scores:")
            for i, conf in enumerate(result['confidence']):
                print(f"  Class {i}: {conf:.4f} ({conf*100:.2f}%)")
            return result
        else:
            error = response.json()
            print(f"\n❌ Error: {error}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running first!")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

def check_api_health(api_url="http://127.0.0.1:5000/health"):
    """
    Check if the API is running
    
    Args:
        api_url (str): URL of the health endpoint
        
    Returns:
        bool: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is healthy")
            print(f"Model: {result.get('model', 'Unknown')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API is not running")
        return False
    except Exception as e:
        print(f"❌ Error checking API health: {e}")
        return False

if __name__ == "__main__":
    """
    Command line usage
    """
    print("=" * 60)
    print("MobileNet Prediction API Client")
    print("=" * 60)
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("\nUsage: python api_call.py <image_path> [api_url]")
        print("\nExample:")
        print("  python api_call.py test_image.jpg")
        print("  python api_call.py test_image.jpg http://127.0.0.1:5000/predict")
        print("\nFirst, make sure to:")
        print("  1. Train a model using the Streamlit app")
        print("  2. Activate the API in the Streamlit app")
        print("  3. Then run this script with your test image")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5000/predict"
    
    # Extract base URL for health check
    base_url = api_url.rsplit('/', 1)[0]
    health_url = f"{base_url}/health"
    
    # Check API health first
    print("\n1. Checking API health...")
    if not check_api_health(health_url):
        print("\n⚠️ Warning: API might not be running")
        print("Please activate the API in the Streamlit app first")
        print("Attempting prediction anyway...\n")
    
    # Make prediction
    print("\n2. Making prediction...")
    result = predict_image(image_path, api_url)
    
    if result:
        print("\n" + "=" * 60)
        print("Prediction complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Prediction failed!")
        print("=" * 60)