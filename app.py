from flask import Flask, render_template, request, jsonify
import pickle
from test import TextToNum  # Ensure this file and class exist

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            msg = request.form.get("message")  # Get input from form
            if not msg:
                return jsonify({"error": "No message provided"})

            print("User Input:", msg)

            # Process input text
            ob = TextToNum(msg)
            ob.cleaner()
            ob.token()
            ob.removeStop()
            st = ob.stemme()

            with open("vectorizer.pickle", "rb") as vcfile:
                vc = pickle.load(vcfile)

            stvc = " ".join(st)
            data = vc.transform([stvc])  # Transform input for model

            with open("model.pickle", "rb") as mdfile:
                model = pickle.load(mdfile)

            pred = model.predict(data)  # Predict sentiment
            return jsonify({"result": str(pred[0])})

        except Exception as e:
            return jsonify({"error": str(e)})  # Return error as JSON

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)