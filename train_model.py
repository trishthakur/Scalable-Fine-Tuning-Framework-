"""
Model training and evaluation functions
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

def train_model(model_version, learning_rate, batch_size, epochs, data, num_classes):
    """
    Train a MobileNet model
    
    Args:
        model_version (str): "MobileNet" or "MobileNetV2"
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        data (tuple): (X_train, y_train) training data
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Trained model
    """
    X_train, y_train = data

    print(f"Training {model_version} with:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Classes: {num_classes}")
    print(f"  Training samples: {len(X_train)}")

    # Load the base model
    base_model = (
        MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        if model_version == "MobileNet"
        else MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    )
    base_model.trainable = False
    
    # Define the model architecture
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Train the model
    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    print("Training complete!")
    return model

def evaluate_model(model, data):
    """
    Evaluate a trained model
    
    Args:
        model (tf.keras.Model): Trained model
        data (tuple): (X_test, y_test) test data
        
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    X_test, y_test = data

    print(f"Evaluating model on {len(X_test)} test samples...")

    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Calculate precision, recall, and F1 score
    precision1 = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Evaluation complete!")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision1:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    return accuracy, precision1, recall, f1