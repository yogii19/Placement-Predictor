from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        cgpa = float(request.form["cgpa"])
        internships = int(request.form["internships"])
        projects = int(request.form["projects"])
        communication = int(request.form["communication"])

        # Prepare input data
        data = np.array([[cgpa, internships, projects, communication]])

        # Prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        if prediction == 1:
            result = f"🎉 You are likely to be PLACED (Probability: {probability:.2f})"
        else:
            result = f"⚠️ You are NOT likely to be placed (Probability: {probability:.2f})"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))


# IMPORTANT for deployment (Render)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)