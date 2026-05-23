# Food Calorie Estimator 🍕

A machine learning project that predicts the type of food from an uploaded image and estimates its calories using a custom Convolutional Neural Network trained with TensorFlow. It features a simple and interactive user interface built with Streamlit.

## Features
- **Image Upload:** Upload pictures of food (`.jpg`, `.jpeg`, `.png`).
- **Classification:** Uses a trained deep learning model (`food_model.h5`) to classify the food.
- **Calorie Estimation:** Provides the estimated calories per 100g for the identified food item.

## Project Structure
- `app.py`: The main Streamlit web application.
- `dl_food_classification.ipynb`: Jupyter Notebook containing the data preparation, model architecture, and training process.
- `food_model.h5`: The trained Keras/TensorFlow model.
- `class_labels.json`: A mapping of prediction indices to human-readable food names.
- `dataset/`: Directory containing the training data.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd "Food Classification"
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

This will start a local server, and you can view the application in your browser (usually at `http://localhost:8501`).
