from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)

# Load dataset
ice = pd.read_csv('ice_cream_temp_sales (1).csv')

# Split input and output
X = ice[['Temperature_C']]
y = ice['IceCream_Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Simple HTML template
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Ice Cream Sales Prediction</title>
</head>
<body style="font-family: Arial; text-align: center;">
    <h2>Ice Cream Sales Prediction 🍦</h2>
    <form method="post">
        <label>Enter Temperature (°C):</label><br><br>
        <input type="number" step="0.1" name="temp" required>
        <br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <h3>Predicted Ice Cream Sales: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        temp = float(request.form['temp'])
        prediction = model.predict([[temp]])[0]
        prediction = round(prediction, 2)

    return render_template_string(html, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
