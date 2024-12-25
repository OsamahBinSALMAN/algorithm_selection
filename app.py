from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS kütüphanesini ekledik
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

"""
@app.route('/evaluate', methods=['POST'])

def evaluate():
    try:
        # Get file and dropdown values
        file = request.json['file']
        problem_type = request.json['problem_type']
        dataset_area = request.json['dataset_area']
        return jsonify({"status": "error", "message": file.filename})

        # Validate file type
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            return jsonify({"status": "error", "message": "Invalid file format. Only CSV or XLSX files are allowed."})

        # Read file into a DataFrame
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)

        # Example processing based on dropdown values
        result = {
            "problem_type": problem_type,
            "dataset_area": dataset_area,
            "num_rows": len(df),
            "num_columns": len(df.columns)
        }

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": "hahahaahha"})

if __name__ == '__main__':
    app.run(debug=True)


"""

# Folder to store uploaded files temporarily
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



