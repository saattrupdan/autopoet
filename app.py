from flask import Flask, request, jsonify, render_template
import numpy as np
from syllablecounter import load_model

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    inputs = list(request.form.values())[0]
    syls = model.predict(inputs)
    return render_template('index.html', pred_text = f'{syls} syllables')

@app.route('/results', methods = ['POST'])
def results():
    data = request.get_json(force = True)
    preds = [model.predict(doc) for doc in data.values()]
    return jsonify(preds)

if __name__ == "__main__":
    app.run()
