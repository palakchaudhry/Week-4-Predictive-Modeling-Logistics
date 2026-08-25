import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the logistics dataset
df = pd.read_excel("Week3_Hypothetical_Logistics_Dataset.xlsx")

# Features and target
X = df[
    [
        "Distance_km",
        "Shipment_Volume_kg",
        "Transportation_Cost_INR",
        "Vehicle_Type",
        "Region",
        "Delivery_Status"
    ]
]

y = df["Delivery_Time_days"]

# Identify categorical and numerical columns
categorical_features = [
    "Vehicle_Type",
    "Region",
    "Delivery_Status"
]

numerical_features = [
    "Distance_km",
    "Shipment_Volume_kg",
    "Transportation_Cost_INR"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

print("Model training completed successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R-squared:", round(r2, 2))


# Optimization Analysis

avg_distance = df["Distance_km"].mean()
avg_cost = df["Transportation_Cost_INR"].mean()
avg_delivery_time = df["Delivery_Time_days"].mean()

print("Optimization Analysis")
print("---------------------")
print("Average Distance (km):", round(avg_distance, 2))
print(
    "Average Transportation Cost (INR):",
    round(avg_cost, 2)
)
print(
    "Average Delivery Time (days):",
    round(avg_delivery_time, 2)
)

print("\nOptimization Recommendation:")
print(
    "Prioritize shorter delivery routes to reduce "
    "transportation cost and delivery time."
)
