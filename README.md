# Intelligent Stethoscope – README  

## Introduction  
**Intelligent Stethoscope** is a Flask-based web application designed to predict respiratory diseases using deep learning. The system processes audio recordings of breath sounds and classifies them into six categories:  
- Bronchiectasis  
- Bronchiolitis  
- COPD (Chronic Obstructive Pulmonary Disease)  
- Healthy  
- Pneumonia  
- URTI (Upper Respiratory Tract Infection)  

The application utilizes **Mel-Frequency Cepstral Coefficients (MFCCs)** to extract features from the audio and predicts the disease using a **pre-trained TensorFlow model**.

## Features  
- Upload audio files of breath sounds  
- Extract MFCC features for analysis  
- Predict respiratory diseases using a deep learning model  
- Display results in a user-friendly web interface  

## Installation  

### Prerequisites  
Ensure you have **Python 3.8+** installed along with the following dependencies:  
- Flask  
- TensorFlow  
- Librosa  
- NumPy  

### Setup  
1. Clone this repository:  
   ```
   git clone https://github.com/saurabh-G07/Intelligent-Stethoscope.git  
   cd Intelligent-Stethoscope  
   ```  
2. Install dependencies:  
   ```
   pip install -r requirements.txt  
   ```  
3. Run the application:  
   ```
   python app.py  
   ```  
4. Open the application in your browser:  
   ```
   http://127.0.0.1:5000/
   ```  

## Usage  
1. Upload a **.wav** audio file of breath sounds.  
2. Click **Submit** to analyze the recording.  
3. The predicted disease will be displayed on the screen.  

## Model  
- The model used for prediction is a deep learning **Keras model** (`resp_model_300.h5`).  
- It was trained using audio datasets with labeled respiratory conditions.  

## File Structure  
```
Intelligent-Stethoscope/  
│── templates/  
│   ├── index.html       # Web interface  
│── app.py               # Flask application  
│── resp_model_300.h5    # Pre-trained model  
│── requirements.txt     # Dependencies  
│── static/              # (Optional) for styling or additional assets  
```  

## Future Enhancements  
- Improve model accuracy with more training data  
- Extend support for additional respiratory diseases  
- Implement a mobile-friendly UI  

## License  
This project is licensed under the **MIT License**.  

## Contributors  
Developed by **Saurabh Mahadev Ghundre**.  
