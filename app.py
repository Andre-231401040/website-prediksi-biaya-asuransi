from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def prediction():
    age = request.form["umur"]
    bmi = request.form["bmi"]
    children = request.form["jumlah-anak"]
    sex = request.form["jenis-kelamin"]
    region = request.form["wilayah"]
    smoker = request.form["perokok"]
    data = np.array([[age, bmi, children, sex, region, smoker]])
    predictionResult = model.predict(data)
    return render_template("index.html", result=predictionResult[0])

if __name__ == "__main__":
    app.run(debug=True)

