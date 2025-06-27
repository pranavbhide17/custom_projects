import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Simulated dataset
np.random.seed(42)
data = {
    'bedrooms': np.random.randint(1, 6, 200),
    'bathrooms': np.random.randint(1, 4, 200),
    'square_feet': np.random.randint(500, 4000, 200),
    'location': np.random.randint(1, 5, 200),  # Encoded locations (1: Suburbs, 2: City, etc.)
    'year_built': np.random.randint(1950, 2021, 200),
    'price': np.random.randint(100000, 1000000, 200)  # Random prices
}

df = pd.DataFrame(data)

# Feature engineering
df['house_age'] = 2025 - df['year_built']  # Assuming the current year is 2025
df['price_per_sqft'] = df['price'] / df['square_feet']
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)  # Avoid division by zero

# Select features for training
X = df[['bedrooms', 'bathrooms', 'square_feet', 'location', 'house_age', 'bed_bath_ratio']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('enhanced_house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training completed and saved!")
