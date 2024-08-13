from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import multLinRegres


app = Flask(__name__)

x_values = multLinRegres.x_train

w = np.array([603239.65, 164783.98, -18270.65, 131150.53, 20676.85])
b = 293581.4865476389

def predict(x_values, w, b, input_features):
    # normalizes the inputs to make sure w and b perform correct operations
    minVals = np.min(x_values, axis=0)
    maxVals = np.max(x_values, axis=0)
    normalized_features = (input_features - minVals) / (maxVals - minVals)

    price = np.dot(w, normalized_features) + b
    return price


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    price = None
    if request.method == 'POST':
        sqft = float(request.form.get("sqft"))
        NumBeds = float(request.form.get("NumBeds"))
        NumBaths = float(request.form.get("NumBaths"))
        lotSize = float(request.form.get("lotSize"))
        age = float(request.form.get("age"))

        features = np.array([sqft, NumBeds, NumBaths, lotSize, age])

        price = predict(x_values, w, b, features)
        return render_template('index.html', price=round(price,2))
    return render_template('index.html', price=price)

@app.route('/reset', methods=['GET'])
def reset():
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)