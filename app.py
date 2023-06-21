import pickle
import json
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='template')

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve user input as a JSON string
        input_str = request.form.get("query")

        # Parse the JSON string into a dictionary
        input_data = json.loads(input_str)

        # Validate input
        if not validate_input(input_data):
            error_msg = "Invalid input. Please provide all required fields."
            return render_template('home.html', error_msg=error_msg)

        # Create a DataFrame from the input dictionary
        input_df = pd.DataFrame([input_data])

        # Load the trained model
        model = pickle.load(open("PumpDiagnosticModel1.pkl", "rb"))

        # Make a prediction and calculate the probability
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        # Determine the pump condition and confidence
        pump_conditions = ["healthy", "inception", "developing", "mild", "severe", "unstable"]
        condition = pump_conditions[prediction[0]]
        confidence = probability[0] * 100

        # Render the results on the home page
        output1 = f"The pump is in {condition} condition"
        output2 = f"Confidence: {confidence:.2f}%"

        return render_template('home.html', output1=output1, output2=output2)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return render_template('home.html', error_msg=error_msg)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

def validate_input(input_data):
    # Check if all required fields are present
    required_fields = ["oneNormAcc1", "meanAcc1", "maxAcc1", "kurAcc1", "varAcc1", "oneNormAcc2", "meanAcc2", "maxAcc2",
                       "kurAcc2", "varAcc2", "oneNormSP", "meanSP", "maxSP", "kurSP", "varSP", "oneNormAcc1f",
                       "meanAcc1f", "maxAcc1f", "kurAcc1f", "varAcc1f", "oneNormAcc2f", "meanAcc2f", "maxAcc2f",
                       "kurAcc2f", "varAcc2f", "maxSPf", "kurSPf", "varSPf", "oneNormSPf", "meanSPf"]
    for field in required_fields:
        if field not in input_data:
            return False
    return True

if __name__ == "__main__":
    app.run()
