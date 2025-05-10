

# Machine Condition Prediction using Random Forest

**Submitted by:**
**Name:** Tamilselvan .J
**Year:** 2nd Year
**Department:** Mechanical Engineering
**Course:** Data Analysis in Mechanical Engineering
**College:** ARM College of Engineering & Technology

---

## Project Overview

This project is focused on predicting the condition of a mechanical machine using a machine learning model. The model used here is a **Random Forest Classifier**, which takes input data such as temperature, vibration, oil quality, and RPM to predict whether a machine is in a normal or faulty condition.

---

## Software Requirements

Before running the prediction program, make sure you install the required Python libraries. You can do that by running:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages mentioned in the `requirements.txt` file.

---

## Files Needed for Prediction

To successfully run the prediction, the following files must be in your working directory:

* **`random_forest_model.pkl`** – This is the trained Random Forest model used to make predictions.
* **`scaler.pkl`** – This file contains the scaler used to normalize input features before prediction.
* **`selected_features.pkl`** – This file helps to maintain the exact order of features the model was trained on.

Make sure these files are present in the same folder as your Python script.

---

## Step-by-Step Explanation: How Prediction Works

1. **Loading the Required Files**

   * The model is loaded using `joblib.load('random_forest_model.pkl')`.
   * The scaler used for preprocessing is also loaded.
   * The list of selected features is loaded to make sure inputs are correctly aligned.

2. **Input Data Preparation**

   * A single row of data is created in a pandas DataFrame.
   * This row must contain all the required features used during model training.

3. **Data Preprocessing**

   * The scaler is applied to the input to match the training format.

4. **Making Predictions**

   * The `.predict()` function is used to find out if the machine is normal or faulty.
   * The `.predict_proba()` function gives the confidence level for each prediction.

---

## Example Python Code to Make Predictions

You can use the following code as a starting point to make predictions:

```python
import joblib
import pandas as pd

# Load the trained model and related files
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input data
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange data in the correct feature order
new_data = new_data[selected_features]

# Apply scaling
scaled_data = scaler.transform(new_data)

# Predict machine condition
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

## Important Points to Remember

* The input data must match the features used during training, including their names and order.
* The values should be within realistic ranges. For example, temperature or RPM should not be unusually high or low.
* Always use the same scaler and feature list that was used during the training of the model.

---

## If You Want to Retrain the Model

To update or retrain the model with new data:

* Follow the same preprocessing steps.
* Ensure the features and scaling remain consistent.
* After training, save the updated model, scaler, and feature list using `joblib`.

---

## Real-World Applications

* This kind of model can be useful in factories for monitoring the health of machines.
* It can be applied in predictive maintenance to avoid machine breakdown.
* Can also be integrated into IoT devices that collect sensor data.

