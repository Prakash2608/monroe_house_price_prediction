from flask import Flask, render_template, jsonify, request

from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.logger.logging import  logging
from src.exceptions.exception import customexception

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if  request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            location = str(request.form.get("location")),
            total_sqft = float(request.form.get("total_sqft")),
            bath = float(request.form.get("bath")),
            bhk = int(request.form.get("bhk"))
        )
        
        final_data = data.get_data_as_dataframe()
        
        predict_pipeline = PredictPipeline()
        
        pred = predict_pipeline.predict(final_data)
        
        result = round(pred[0],2)
        price = result * 100000
        
        return render_template("result.html", final_result = price)
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0')