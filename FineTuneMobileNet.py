"""
Streamlit application for MobileNet fine-tuning
Local deployment version
"""
import streamlit as st
import tensorflow as tf
from train_model import train_model, evaluate_model
from metrics_db import save_model_metrics
from local_storage import save_model_locally, list_saved_models
from model_api import activate_api
import pickle
import os
from PIL import Image
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split
from config import get_connection

st.set_page_config(page_title="MobileNet Fine-Tuning", layout="wide")

st.title(" MobileNet Model Training System")
st.markdown("### Fine-tune MobileNet models on your custom image dataset")

# Sidebar for model information
with st.sidebar:
    st.header(" Saved Models")
    saved_models = list_saved_models()
    if saved_models:
        st.write(f"Total models: {len(saved_models)}")
        for model in saved_models:
            st.text(f"• {model}")
    else:
        st.info("No models saved yet")
    
    st.markdown("---")
    st.markdown("### Quick Guide")
    st.markdown("""
    1. Enter model name
    2. Select version & parameters
    3. Upload dataset (zip)
    4. Train model
    5. Activate API for predictions
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Model Configuration")
    
    # User Input: Model Name
    model_name = st.text_input(
        "Model Name", 
        placeholder="e.g., my_classifier_v1",
        help="Choose a unique name for your model"
    )
    
    if not model_name:
        st.warning(" Please enter a model name before proceeding.")
    elif model_name in saved_models:
        st.warning(f" Model '{model_name}' already exists. Choose a different name or it will be overwritten.")

with col2:
    st.subheader("Model Version")
    model_version = st.selectbox(
        "Select MobileNet Version", 
        ["MobileNet", "MobileNetV2"],
        help="MobileNet is lighter, MobileNetV2 is more accurate"
    )

# Hyperparameters
st.subheader(" Hyperparameters")
col3, col4, col5 = st.columns(3)

with col3:
    learning_rate = st.slider(
        "Learning Rate", 
        0.0001, 0.1, 0.001, 
        format="%.4f",
        help="Controls how much to adjust the model in response to error"
    )

with col4:
    batch_size = st.slider(
        "Batch Size", 
        8, 64, 32,
        help="Number of samples processed before model update"
    )

with col5:
    epochs = st.slider(
        "Epochs", 
        1, 50, 10,
        help="Number of complete passes through the training dataset"
    )

# Dataset Upload
st.subheader(" Dataset Upload")
st.info("Upload a ZIP file containing images organized in folders by class")

uploaded_file = st.file_uploader(
    "Choose a ZIP file", 
    type=["zip"],
    help="Max size: 200MB. Organize images in folders by class name"
)

train_data = None
test_data = None
num_classes = 0

if uploaded_file is not None:
    with st.spinner("Processing dataset..."):
        try:
            # Extract images from the zip file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall("/tmp/images")
                image_files = [f for f in zip_ref.namelist() if not f.startswith('__MACOSX')]
                
            st.success(f" Extracted {len(image_files)} files")

            # Process the images
            image_data = []
            labels = []
            label_names = {}
            current_label = 0

            for image_file in image_files:
                # Skip directories and hidden files
                if image_file.endswith('/') or '/.DS_Store' in image_file:
                    continue
                    
                try:
                    # Extract class name from folder structure
                    parts = image_file.split('/')
                    if len(parts) > 1:
                        class_name = parts[-2]
                    else:
                        class_name = "unknown"
                    
                    # Assign numeric label
                    if class_name not in label_names:
                        label_names[class_name] = current_label
                        current_label += 1
                    
                    label = label_names[class_name]
                    
                    # Load and process image
                    img = Image.open(os.path.join("/tmp/images", image_file))
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    
                    image_data.append(img_array)
                    labels.append(label)
                    
                except Exception as e:
                    st.warning(f"Could not process {image_file}: {str(e)}")

            if len(image_data) == 0:
                st.error("No valid images found in the ZIP file")
            else:
                # Convert to numpy arrays
                image_data = np.array(image_data)
                labels = np.array(labels)
                num_classes = len(label_names)

                # Display dataset info
                col6, col7, col8 = st.columns(3)
                with col6:
                    st.metric("Total Images", len(image_data))
                with col7:
                    st.metric("Number of Classes", num_classes)
                with col8:
                    st.metric("Image Size", "224x224")
                
                # Show class distribution
                st.write("**Class Distribution:**")
                class_counts = {}
                for name, idx in label_names.items():
                    count = np.sum(labels == idx)
                    class_counts[name] = count
                
                for name, count in sorted(class_counts.items()):
                    st.text(f"  {name}: {count} images")

                # Split the data
                train_data, test_data = train_test_split(
                    list(zip(image_data, labels)), 
                    test_size=0.2, 
                    random_state=42
                )
                
                st.success(f" Training: {len(train_data)} images | Testing: {len(test_data)} images")

        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")

# Training section
st.markdown("---")
col9, col10 = st.columns(2)

with col9:
    train_button = st.button(" Train Model", type="primary", use_container_width=True)

with col10:
    api_button = st.button(" Activate API", use_container_width=True)

# Start training
if train_button:
    if not model_name:
        st.error(" Please enter a model name before training.")
    elif uploaded_file is None or train_data is None or test_data is None:
        st.error(" Please upload a valid dataset first.")
    elif num_classes < 2:
        st.error(" Dataset must contain at least 2 classes.")
    else:
        try:
            # Separate images and labels
            X_train, y_train = zip(*train_data)
            X_test, y_test = zip(*test_data)

            # Convert to numpy arrays
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # Training progress
            with st.spinner(" Training model... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing model...")
                progress_bar.progress(10)
                
                # Train model
                model = train_model(
                    model_version, 
                    learning_rate, 
                    batch_size, 
                    epochs, 
                    (X_train, y_train),
                    num_classes
                )
                
                progress_bar.progress(70)
                status_text.text("Evaluating model...")

                # Evaluate model
                accuracy, precision1, recall, f1 = evaluate_model(model, (X_test, y_test))
                
                progress_bar.progress(90)
                status_text.text("Saving model...")

                # Display results
                st.success(" Training Complete!")
                
                st.subheader(" Test Metrics")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with metric_col2:
                    st.metric("Precision", f"{precision1:.2%}")
                with metric_col3:
                    st.metric("Recall", f"{recall:.2%}")
                with metric_col4:
                    st.metric("F1 Score", f"{f1:.2%}")

                # Save metrics to database
                try:
                    save_model_metrics(model_name, accuracy, precision1, recall, f1)
                    st.success(" Metrics saved to database")
                except Exception as e:
                    st.error(f" Error saving metrics: {e}")

                # Save model locally
                model_path = f"{model_name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                
                save_model_locally(model_path, model_name)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                st.success(f" Model saved as '{model_name}.pkl'")
                
                # Cleanup temp file
                if os.path.exists(model_path):
                    os.remove(model_path)
                    
        except Exception as e:
            st.error(f" Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Activate API
if api_button:
    if not model_name:
        st.error(" Please enter a model name to activate the API.")
    elif model_name not in saved_models:
        st.error(f" Model '{model_name}' not found. Please train it first.")
    else:
        try:
            with st.spinner("Starting API server..."):
                api_url = activate_api(f"./models/{model_name}.pkl")
                
            st.success(" API is running!")
            st.info(f"**API URL:** `{api_url}`")
            
            st.markdown("### How to use the API:")
            st.code(f"""
import requests

url = "{api_url}"
image_path = "your_image.jpg"

with open(image_path, 'rb') as img:
    files = {{'file': img}}
    response = requests.post(url, files=files)
    
result = response.json()
print("Predicted Class:", result['predicted_class'])
print("Confidence:", result['confidence'])
            """, language="python")
            
            st.warning(" Note: The API runs in a background thread and will stop when you close this Streamlit app.")
            
        except Exception as e:
            st.error(f" Error activating API: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>MobileNet Fine-Tuning System | Local Deployment Version</p>
    <p><small>Original GCP implementation by Kalyani Jaware & Trishala Thakur</small></p>
</div>
""", unsafe_allow_html=True)