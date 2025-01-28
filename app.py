from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__, template_folder='template')

model = joblib.load('iris_model.pkl')

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)
    flower_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result = flower_map[prediction[0]]

    return render_template('index.html', prediction_text='The flower is {}'.format(result))

@app.route('/favicon.ico')
def favicon():
    return '' , 204


if __name__ == '__main__':
    app.run(debug=True)