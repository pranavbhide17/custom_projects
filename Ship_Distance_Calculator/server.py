from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('enhanced_house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Calculate derived features from inputs
        house_age = 2025 - data['year_built']
        bed_bath_ratio = data['bedrooms'] / (data['bathrooms'] + 1)

        # Extract input features
        features = [
            data['bedrooms'],
            data['bathrooms'],
            data['square_feet'],
            data['location'],
            house_age,
            bed_bath_ratio
        ]

        # Make prediction
        prediction = model.predict([features])[0]
        return jsonify({'predicted_price': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
