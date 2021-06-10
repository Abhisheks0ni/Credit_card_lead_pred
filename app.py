import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load objects 

stk1 = pickle.load(open('stk1', 'rb'))
feat_map = pickle.load(open('feat_map', 'rb'))
fe_enc = pickle.load(open('fe_enc', 'rb'))
fe_enc_single = pickle.load(open('fe_enc_single', 'rb'))
comb_list = pickle.load(open('comb_list', 'rb'))

# Categorical and numerical features

num_cols = ['Age', 'Avg_Account_Balance', 'Vintage']
cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product'
            , 'Is_Active']


def frequency_encoding(column_name,output_column_name,df,fe_pol):
    df[output_column_name] = df[column_name].apply(lambda x : fe_pol.at[x])

def genrate_df(input_):
    val = list(input_.values())
    key = list(input_.keys())
    df = pd.DataFrame([val],columns=key) 
    return df

def preprocess(df):
    # convert numeric cols
    
    for i in num_cols:
        df[i] = df[i].astype('int64')
        
    # Frequency encoding mapping    
    names_single = []
    for i in range(len(fe_enc_single)):
        names_single.append(fe_enc_single[i].index.name)
    
    names = []
    for i in range(len(fe_enc)):
        names.append(fe_enc[i].index.name)
    
    
    # For pair of features
    for i in comb_list:
        j = names.index(f'{i[0]}_{i[1]}') 
        df[f'{i[0]}_{i[1]}']=df[i[0]].astype(str)+'_'+df[i[1]].astype(str)
        frequency_encoding(f'{i[0]}_{i[1]}',f'{i[0]}_{i[1]}',df,fe_enc[j])
        
    # For individual feature    
    for i in list(cat_cols):  
        if i!='Credit_Product':
            j = names_single.index(f'{i}') 
            df[str(i)+'fe']=df[i].astype(str)
            frequency_encoding(i,str(i)+'fe',df,fe_enc_single[j])
        
    # Encoding
    for i in cat_cols:
        map_ = feat_map[i]
        k = df[i].values[0] 
        df[i]=map_[str(k)]
        
    return df

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Genrate df
    
    input_= request.form.to_dict()
    
    # Genrate dataframe
    df = genrate_df(input_)
    
    # Preprocess dataframe
    df = preprocess(df)
   
    # Predict
    prediction = stk1.predict_proba(df.values)[:,1]
    prediction = np.round(prediction*100, 2)
    prediction = ' '.join(map(str, prediction))
    pred = 'There are '+ prediction +' % chances that customer will buy recommended credit card.'
    return render_template("home.html",prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
