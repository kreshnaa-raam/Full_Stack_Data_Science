import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('customf.pkl', 'rb'))


def preprocess_features(X):
    """Add any required feature preprocessing here, if it's not handled by the pickled model"""

    # Preprocess FlightDate
    X['FlightDate'] = pd.to_datetime(X['FlightDate'])
    X['Day_of_Week'] = X['FlightDate'].dt.day_name
    X['day'] = X['FlightDate'].dt.day
    X['month'] = X['FlightDate'].dt.month
    X['year'] = X['FlightDate'].dt.year
    X = X.drop('FlightDate', axis=1)
    #X = X.drop('DepTime', axis=1)

    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [list(int_features)]
    final_features = pd.DataFrame(final_features, columns=['FlightDate', 'UniqueCarrier',
                                                           'Origin', 'Dest', 'Distance'])

    final_features = preprocess_features(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 0:
        return render_template('index.html', prediction_text = 'Flight is not Delayed')
    else:
        return render_template('index.html', prediction_text = 'Flight is Delayed')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json()
    data = pd.DataFrame.from_dict(data)
    data = preprocess_features(data)
    prediction = model.predict(data)
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 5000)