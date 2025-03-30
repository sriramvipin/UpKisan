import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('crop_production.csv')  # Use '/' orṇ double backslashes '\\'
# Data preprocessing
# Assuming the dataset has columns: ['Location', 'Month', 'Land (hectares)', 'Soil Type', 'Crop', 'Production (tons)', 'Profit']
df = pd.get_dummies(df, columns=['District_Name', 'Crop_Year', 'Season', 'Crop'])

# Define features and target
X = df.drop('Profit', axis=1)
y = df['Profit']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Prediction function
def predict_profit(location, month, land_size, soil_type, crop):
    # Create input dictionary ensuring all required columns exist
    input_data = pd.DataFrame({
        'Land (hectares)': [land_size],
        **{f'Location_{location}': 1 if f'Location_{location}' in X.columns else 0},
        **{f'Month_{month}': 1 if f'Month_{month}' in X.columns else 0},
        **{f'Soil Type_{soil_type}': 1 if f'Soil Type_{soil_type}' in X.columns else 0},
        **{f'Crop_{crop}': 1 if f'Crop_{crop}' in X.columns else 0}
    }, index=[0]).reindex(columns=X.columns, fill_value=0)  # Ensure column order
    
    return model.predict(input_data)[0]

# Example prediction
print(predict_profit('Location_A', 'January', 5, 'Loamy', 'Wheat'))
