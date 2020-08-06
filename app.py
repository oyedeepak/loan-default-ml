import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rfmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    feature_names = ['State', 'NAICS', 'NoEmp', 'NewExist', 'RevLineCr', 'LowDoc', 'GrAppv', 'SBA_Appv', 'Term']
    df = pd.DataFrame(final_features, columns = feature_names)
    prediction = model.predict(df)
    if prediction==0.0:
        output="PIF"
    else:
        output="CHGOFF"

    return render_template('index.html', prediction_text='Loan is {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)