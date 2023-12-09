from flask import Flask, request, jsonify
from flask_cors import CORS
import MLLib

app = Flask("Laptop Price Predictor")
CORS(app)


@app.route("/")
def home():
    return "Welcome to home page"


@app.route("/api/predict_price", methods=['POST'])
def predict_price():
    '''
        Takes the laptop configuration
    :return:
        Returns a list of 3 similar laptop configurations and the input laptop configuration with predicted price
    '''
    if request.is_json:
        data = request.get_json()['formdata']
        print(data)
        laptops = MLLib.predict_price(data)

        result = laptops
        return jsonify(result), 200
    else:
        return jsonify({"error": "Invalid data format"}), 400


@app.route("/api/get_brand_and_price_chart", methods=['GET'])
def get_brand_and_price_chart():
    '''
    :return:
        Returns a list of laptop manufacturers with their average laptop price
    '''
    data = MLLib.get_brand_price_chart_data()
    chart_data = []
    i = 0
    for brand, value in data.items():
        chart_data.append({"id": i, "brand_name": brand, "value": value})
        i += 1

    if chart_data:
        return jsonify(chart_data), 200
    else:
        return jsonify({"error": "Cannot get chart data"}), 400


if __name__ == '__main__':
    app.run()
