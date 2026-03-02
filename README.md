# skin-cancer-prediction
# 🧠 Skin Cancer Detection using CNN + FastAPI

A Deep Learning–based Skin Cancer Classification system built using **TensorFlow (CNN)** and deployed using **FastAPI**.  
This project uses the **HAM10000 dataset** to classify different types of skin lesions.

---

## 🚀 Project Overview

This system:

- 📊 Trains a Convolutional Neural Network (CNN) on dermatoscopic images  
- 🏷 Encodes lesion labels using `LabelEncoder`  
- 💾 Saves trained model in `.keras` format  
- 🔁 Saves label encoder using `pickle`  
- 🌐 Deploys prediction API using FastAPI  
- 📤 Accepts image uploads and returns predicted skin disease type  

---

## 🗂 Dataset

**HAM10000 – Human Against Machine with 10,000 training images**

- Metadata: `HAM10000_metadata.csv`
- Images Folder: `HAM10000_images_part_1/`
- Image Size Used: `64 x 64`
- Normalization: Pixel values scaled to `[0,1]`

---

## 🧠 CNN Model Architecture


Input Layer (64x64x3)

Conv2D (32 filters, 3x3) → ReLU
MaxPooling (2x2)

Conv2D (64 filters, 3x3) → ReLU
MaxPooling (2x2)

Flatten

Dense (128 neurons) → ReLU
Dropout (0.5)

Output Layer (Softmax - Multi-class)


---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy & Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Matplotlib
- Pickle

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/skin-cancer-cnn.git
cd skin-cancer-cnn
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt is not created, install manually:

pip install tensorflow fastapi uvicorn numpy pandas opencv-python scikit-learn matplotlib pillow
🏋️ Model Training

Run the Jupyter Notebook:

skin_cancer_pred.ipynb

After training, it will generate:

skin_cancer_cnn_model.keras
label_encoder.pkl
🌐 Running FastAPI Server

Start the API server:

uvicorn main:app --reload

If your file name is different, replace main with your filename.

Server will run at:

http://127.0.0.1:8000

Swagger Documentation:

http://127.0.0.1:8000/docs
🔍 API Endpoints
✅ Health Check
GET /

Response:

{
  "message": "Skin Cancer Prediction API is live!"
}
📤 Predict Skin Cancer
POST /predict

Input: Image file (JPG)

Response:

{
  "prediction": "melanoma"
}
📊 Training Visualization

The project visualizes:

Training Accuracy

Validation Accuracy

Training Loss

Validation Loss

Using Matplotlib.

📁 Project Structure
skin-cancer-cnn/
│
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
├── skin_cancer_pred.ipynb
├── skin_cancer_cnn_model.keras
├── label_encoder.pkl
├── main.py
├── README.md
🧪 Model Evaluation

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Epochs: 10

Batch Size: 32

Validation Split: 0.1

🎯 Future Improvements

Increase image size (128x128 or 224x224)

Use Transfer Learning (ResNet / EfficientNet)

Add Data Augmentation

Deploy on Cloud (Render / AWS / GCP)

Build Flutter Mobile App frontend

Add PDF medical report generation
