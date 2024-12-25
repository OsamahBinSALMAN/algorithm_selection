from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS kütüphanesini ekledik
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Tüm yollar ve kökenler için CORS'u aktif ediyoruz

# Kaydedilmiş modeli yükle
with open('bagging_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # AJAX ile gelen veriyi al
        input_array = np.array([float(i) for i in data.split(',')]).reshape(1, -1)  # Veriyi işle
        prediction = model.predict(input_array)[0]  # Modelden tahmin sonucu al
        classes=["Argon","Acetone","Methanol"]
        return jsonify({'prediction': classes[prediction]})  # JSON formatında sonucu döndür
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

