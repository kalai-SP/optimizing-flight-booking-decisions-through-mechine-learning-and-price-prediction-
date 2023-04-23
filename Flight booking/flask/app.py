# importing req lib
from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open("model1.pkl", 'rb'))

@app.route('/')
def about():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route("/pred", methods=['POST', 'GET'])
def pred():
    x = [[int(x) for x in request.form.values()]]
    print(x)

    x = np.array(x)
    print(x.shape)

    print(x)
    print(model.predict(x))
    output = model.predict(x)
    return render_template('submit.html', prediction_text=output[0])


if __name__ == "__main__":
    app.run(debug=True)