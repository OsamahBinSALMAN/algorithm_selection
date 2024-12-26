from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS kütüphanesini ekledik
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA
import statistics

warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def LossDataAnaliz(data):
    loss=list(data.isnull().any(axis=1))
    for a,b in enumerate(loss):
        if b:
            data.drop(a, axis=0, inplace=True)
    return data

#One Hot Encoding
def Categorical_Data_Encoding(data):
    enc = LabelEncoder()
    dtype_0=[a1.name for a1 in list(data.dtypes)]
    names=list(data.columns)
    for index,One_type in enumerate(dtype_0):
        if One_type=="object":
            data[names[index]]=enc.fit_transform(data[names[index]])
    return data

def CorrelationCalculator(data,c):
    corr=[]
    for i in range(c):
        Corr_list=list(data.corr()['output'+str(i+1)])
        Corr_list=[n for n in Corr_list if n!=1 and str(n)!="nan"]
        corr.append(Corr_list)
    Corr_list=[d for d1 in corr for d in d1]
    Absoulte_Corr_list=[abs(n) for n in Corr_list if n!=1]
    
    return round(statistics.mean(Corr_list),5),round(statistics.median(Corr_list),5),round(statistics.stdev(Corr_list),5),round(statistics.mean(Absoulte_Corr_list),5),round(statistics.median(Absoulte_Corr_list),5),round(statistics.stdev(Absoulte_Corr_list),5)
  
def Size_reducation(data,c):
    df=data
    out=[]
    for i in range(c):
        df=df.drop(['output'+str(i+1)], axis=1)
        out.append('output'+str(i+1))
    outs=data[out]
    pca = PCA(n_components =1 )
    pca.fit(df)
    data_pca = pca.transform(df)
    data_pca = pd.DataFrame(data_pca,columns=['PC1'])
    return data_pca,outs
    
def tahmin(model,dt):
    input_array = np.array(dt).reshape(1, -1)
    prediction = model.predict(input_array)[0]  # Modelden tahmin sonucu al    
    algorithms = ["Decision Tree","Ridge Regression","Random Forest","Elastic Net",
                 "Stochastic Gradient Descent","AdaBoost","Gradient Boosting",
                 "HistGradientBoosting","Voting Soft","Linear Regression",
                 "Stacking","Lasso Regression","Extra Trees",
                 "Bagging","K-Nearest Neighbors"]
    sorted_algorithms = [x for _, x in sorted(zip(prediction, algorithms))]
    return sorted_algorithms

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    problem_type = request.form['problem_type']
    if problem_type == '':
        return jsonify({"status": "error", "message": "No problem type file"}), 400
        
    dataset_area = request.form['dataset_area']
    if dataset_area == '':
        return jsonify({"status": "error", "message": "No dataset area file"}), 400

    if file and allowed_file(file.filename) and problem_type=='Regression':
        # Save file to the uploads folder
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Read the file and process it
        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(filepath)
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(filepath)
            else:
                return jsonify({"status": "error", "message": "Failed to Read Data"}), 400

            outputs=len([d for d in data.columns if "output"  in d])  
            if outputs<1:
                return jsonify({"status": "error", "message": "The output variables are not found, so please modify the output variable names to follow a sequential format: outputn, where n is an integer starting from 1 and increments up to the total number of outputs generated. For example, the first output should be named output1, the second output2, and so on, ensuring that each output is labeled with a unique number corresponding to its position in the sequence. This format will help properly track and organize the outputs."}), 400
            
            number_of_featuers=len([d for d in data.columns if "output" not in d])

            if number_of_featuers<1:
                return jsonify({"status": "error", "message": "Input variables not found"}), 400
            
            data=LossDataAnaliz(data)
            number_of_samples=len(data)-1
            if number_of_samples<10:
                return jsonify({"status": "error", "message": "Samples are less than 10"}), 400
            dtype_=[a1.name for a1 in list(data.dtypes)]
            
            if "object" not in dtype_:
                data_type="Numerical"    
            elif (("float64" not in dtype_) and ("int64" not in dtype_)):
                data_type="Categorical"
            else:
                data_type="Both"
            
            data=Categorical_Data_Encoding(data)            
            ortalama,ortanca,std,abs_ortalama,abs_ortanca,abs_std=CorrelationCalculator(data,outputs)
            Tek_boyut_data,outputs_no=Size_reducation(data,outputs)
            resize_Corr_list=0
            for c in range(outputs):
                resize_Corr_list+=round(np.corrcoef(list(Tek_boyut_data["PC1"]),list(outputs_no["output"+str(c+1)]))[0][1],5)
            resize_Corr_list=round(resize_Corr_list/outputs,5)
            
            names=[n for n in list(data.columns) if "output" not in n]
            Wilks,Pilais,Hotelling,Roys=0,0,0,0 
            for c in range(outputs):
                formul=' + '.join(names)+" ~ output"+str(c+1)
                try:
                    
                    Manova= MANOVA.from_formula(formul, data=data).mv_test().summary_frame
                    Wilks+=round(Manova["Value"]["output"+str(c+1)]["Wilks' lambda"],5)
                    Pilais+=round(Manova["Value"]["output"+str(c+1)]["Pillai's trace"],5)
                    Hotelling+=round(Manova["Value"]["output"+str(c+1)]["Hotelling-Lawley trace"],5)
                    Roys+=round(Manova["Value"]["output"+str(c+1)]["Roy's greatest root"],5)
                except:               
                   pass
            Wilks=round(Wilks/outputs,5)
            Pilais=round(Pilais/outputs,5)
            Hotelling=round(Hotelling/outputs,5)
            Roys=round(Roys/outputs,5)

            input_data = {
                "Application Area":dataset_area,
                "Number Of Featuers":number_of_featuers,
                "Number Of Sampels":number_of_samples,
                "Data Type":data_type,
                "Number Of Output":outputs,
                "Mean Correlation":ortalama,
                "Median Correlation":ortanca,
                "Std Correlation":std,
                "Mean Absoulte Correlation":abs_ortalama,
                "Median Absoulte Correlation":abs_ortanca,
                "Std Absoulte Correlation":abs_std,
                "Size Reduction Correlation":resize_Corr_list,
                "MANOVA Wilks Lambda":Wilks,
                "MANOVA Pillais Trace":Pilais,
                "MANOVA Hotelling Lawley Trace":Hotelling,
                "MANOVA Roys Greatest Root":Roys
                
            }

            pred_data=[["Economic","Health","Social","Technology & Engineering"].index(dataset_area),number_of_featuers,number_of_samples,["Both","Numerical"].index(data_type),outputs,abs_ortalama,abs_ortanca,abs_std,ortalama,ortanca,std,resize_Corr_list,Wilks,Pilais,Hotelling,Roys]
            model_names=["Coefficient of Determination","Test Time","Tarining Time"]
            result_data={"Dataset Featuers":input_data}
            for model_name in model_names:
                with open("Regression/"+model_name+'.pkl', 'rb') as file:
                        model = pickle.load(file)

                result_data[model_name]=tahmin(model,pred_data)
            
            # You can perform any necessary data processing here
            # For example, analyzing the columns based on problem type and dataset area
            
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



