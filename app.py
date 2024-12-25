from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS kütüphanesini ekledik
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get file and dropdown values
        file = request['file']
        problem_type = request['problem_type']
        dataset_area = request['dataset_area']

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


