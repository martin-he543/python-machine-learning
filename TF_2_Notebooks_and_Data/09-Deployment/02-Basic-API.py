from flask import Flask, request, jsonify
import numpy as np  
from tensorflow.keras.models import load_model
import joblib


#### THIS IS WHAT WE DO IN POSTMAN ###
# STEP 1: Create New Request
# STEP 2: Select POST
# STEP 3: Type correct URL (http://127.0.0.1:5000/api/flower)
# STEP 4: Select Body
# STEP 5: Select JSON
# STEP 6: Type or Paste in example json request
# STEP 7: Run 02-Basic-API.py to launch server and confirm the site is running
# Step 8: Run API request

def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(flower)
    
    return classes[class_ind][0]


app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

@app.route('/api/flower', methods=['POST'])
def predict_flower():

    content = request.json
    
    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run()