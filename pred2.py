import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_FILE = 'crop_profit_model.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def train_model(csv_path):
    """Train the model using the uploaded CSV data"""
    try:
        # Load the dataset
        data = pd.read_csv(csv_path)
        
        # Assuming the CSV has these columns (adjust based on your actual data)
        # Example columns: 'crop_type', 'area', 'rainfall', 'fertilizer', 'pesticide', 'labour', 'profit'
        X = data[['crop_type', 'area', 'rainfall', 'fertilizer', 'pesticide', 'labour']]
        y = data['profit']
        
        # Convert categorical data (crop_type) to numerical
        X = pd.get_dummies(X, columns=['crop_type'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Model trained with MAE: {mae}")
        
        return model, X.columns.tolist()
    
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None, None

# Global variables for the model and features
model = None
feature_columns = None

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload the training CSV file"""
    global model, feature_columns
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename('crop_production.csv')  # Force the name to be crop.csv
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Train the model
        model, feature_columns = train_model(filepath)
        
        if model is None:
            return jsonify({'error': 'Failed to train model'}), 500
        
        return jsonify({'message': 'File uploaded and model trained successfully',
                       'features': feature_columns}), 200
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/predict', methods=['POST'])
def predict_profit():
    """Endpoint to predict profit based on user inputs"""
    global model, feature_columns
    
    if model is None:
        return jsonify({'error': 'Model not trained yet. Please upload training data first.'}), 400
    
    try:
        # Get user inputs
        data = request.json
        
        # Create a DataFrame with the user input
        input_data = pd.DataFrame({
            'crop_type': [data['crop_type']],
            'area': [float(data['area'])],
            'rainfall': [float(data['rainfall'])],
            'fertilizer': [float(data['fertilizer'])],
            'pesticide': [float(data['pesticide'])],
            'labour': [float(data['labour'])]
        })
        
        # Convert categorical data to numerical (matching training)
        input_data = pd.get_dummies(input_data, columns=['crop_type'])
        
        # Ensure all columns from training are present (fill missing with 0)
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({
            'predicted_profit': float(prediction[0]),
            'units': 'currency (same as training data)'
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/features', methods=['GET'])
def get_required_features():
    """Endpoint to get the required features for prediction"""
    return jsonify({
        'required_inputs': [
            {'name': 'crop_type', 'type': 'string', 'description': 'Type of crop (e.g., wheat, corn)'},
            {'name': 'area', 'type': 'float', 'description': 'Area cultivated in hectares'},
            {'name': 'rainfall', 'type': 'float', 'description': 'Rainfall in mm'},
            {'name': 'fertilizer', 'type': 'float', 'description': 'Fertilizer amount in kg'},
            {'name': 'pesticide', 'type': 'float', 'description': 'Pesticide amount in kg'},
            {'name': 'labour', 'type': 'float', 'description': 'Labour hours'}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True)