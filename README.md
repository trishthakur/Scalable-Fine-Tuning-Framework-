# MobileNet Fine-Tuning System - Local Deployment

End-to-End System for Fine-tuning Machine/Deep Learning Models with MobileNet

## Overview

This project provides a user-friendly interface for training and evaluating MobileNet models on custom image datasets. Originally designed for Google Cloud Platform, this version is adapted for local deployment.

### Features

- **No-Code ML Training**: Train MobileNet models through a web interface
- **Custom Dataset Support**: Upload your own image datasets (zip format)
- **Hyperparameter Tuning**: Adjust learning rate, batch size, and epochs
- **Model Evaluation**: Get accuracy, precision, recall, and F1 scores
- **REST API**: Deploy trained models as APIs for real-time predictions
- **Local Database**: SQLite for storing model metrics

## Architecture

```
┌─────────────────┐
│   Streamlit UI  │
│   (Frontend)    │
└────────┬────────┘
         │
    ┌────▼────────────────────┐
    │  Training Pipeline      │
    │  - Data Preprocessing   │
    │  - Model Training       │
    │  - Evaluation           │
    └────┬────────────────────┘
         │
    ┌────▼────────┐    ┌──────────────┐
    │   SQLite    │    │  Flask API   │
    │  (Metrics)  │    │ (Predictions)│
    └─────────────┘    └──────────────┘
```
## Demo video

https://github.com/user-attachments/assets/52f651f8-3fcd-4f9b-812f-4625ffe39ba4

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended)
- 2GB free disk space

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/trishthakur/Scalable-Fine-Tuning-Framework-
   cd mobilenet-finetuning-local
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python init_db.py
   ```

## Usage

### 1. Start the Streamlit Application

```bash
streamlit run FineTuneMobileNet.py
```

This will open the web interface in your browser (usually at `http://localhost:8501`)

### 2. Train a Model

1. Enter a unique model name
2. Select MobileNet version (MobileNet or MobileNetV2)
3. Adjust hyperparameters:
   - Learning Rate: 0.0001 - 0.1
   - Batch Size: 8 - 64
   - Epochs: 1 - 50
4. Upload a zip file containing your images
   - Supported formats: JPG, PNG
   - Max size: 200MB
   - Organize by folders for different classes
5. Click "Train Model"

### 3. Activate the API

1. After training, click "Activate API"
2. The API will start on `http://127.0.0.1:5000/predict`
3. Use the provided URL for predictions

### 4. Make Predictions

Use the API client script:

```bash
python api_call.py
```

Or make a POST request:

```python
import requests

url = "http://127.0.0.1:5000/predict"
image_path = "path/to/your/image.jpg"

with open(image_path, 'rb') as img:
    files = {'file': img}
    response = requests.post(url, files=files)
    
print(response.json())
```

## Project Structure

```
mobilenet-finetuning-local/
│
├── FineTuneMobileNet.py    # Main Streamlit application
├── train_model.py           # Model training logic
├── model_api.py             # Flask API for predictions
├── api_call.py              # Example API client
├── metrics_db.py            # Database operations
├── config.py                # Configuration settings
├── preprocessing.py         # Data preprocessing utilities
├── init_db.py              # Database initialization
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── PROJECT_REPORT.pdf      # Original GCP implementation report
│
├── models/                 # Saved trained models (.pkl files)
├── data/                   # Training data (ignored in git)
└── database/              # SQLite database file
```

## Dataset Format

Your dataset should be organized as follows:

```
dataset.zip
│
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── classN/
    ├── image1.jpg
    └── ...
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Use smaller datasets
   - Close other applications

2. **API Not Starting**
   - Check if port 5000 is available
   - Try a different port in `config.py`

3. **Model Training Fails**
   - Verify image format (JPG/PNG)
   - Check dataset structure
   - Ensure images are readable

4. **Database Error**
   - Run `python init_db.py` again
   - Check file permissions

### Performance Tips

- **For faster training**: Increase batch size (if memory allows)
- **For better accuracy**: Increase epochs and use more training data
- **For resource constraints**: Use MobileNet (lighter than MobileNetV2)

## Limitations

- **Dataset Size**: Max 200MB zip file
- **Concurrent Users**: Single user at a time
- **Model Storage**: Limited by local disk space
- **API Scalability**: Not designed for production traffic

## Future Enhancements

- Support for additional data types (text, audio)
- User authentication and access control
- Docker containerization
- GPU support detection and utilization
- Asynchronous job queue for training
- Model versioning system

## References

- [MobileNet Documentation](https://keras.io/api/applications/mobilenet/)
- [Original GCP Implementation Report](PROJECT_REPORT.pdf)

## Contributors

- Kalyani Kailas Jaware
- Trishala Thakur

## License

This project is part of an academic assignment for CSCI 4253 - Datacenter Scale Computing.
