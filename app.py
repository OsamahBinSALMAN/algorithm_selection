from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS kütüphanesini ekledik
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    roblem_type = request.form['problem_type']
    if roblem_type == '':
        return jsonify({"status": "error", "message": "No problem type file"}), 400
        
    dataset_area = request.form['dataset_area']
    if dataset_area == '':
        return jsonify({"status": "error", "message": "No dataset area file"}), 400

    if file and allowed_file(file.filename):
        # Save file to the uploads folder
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read the file and process it
        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(filepath)
            
            # You can perform any necessary data processing here
            # For example, analyzing the columns based on problem type and dataset area
            result_data = {
                "dataset_area":dataset_area,
                "roblem_type":roblem_type,
                "columns": data.columns.tolist(),
                "head": data.head().to_dict(orient='records')
            }

            # Prepare a success response
            return jsonify({
                "status": "success",
                "data": result_data,
                "message": "File processed successfully"
            })

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file type. Only CSV or XLSX files are allowed."}), 400

if __name__ == '__main__':
    app.run(debug=True)



