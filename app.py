from flask import Flask,request, url_for, redirect, render_template
import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np

from pydantic import BaseModel

class disease(BaseModel):
    age:int
    sex:int
    cp:int
    trestbps:int
    chol:int
    fbs:int
    restecg:int
    thalach:int
    exang:int
    oldpeak:int
    slope:int
    ca:int
    thal:int
    
    
app=FastAPI()

mod=pickle.load(open('model.pkl','rb'))

@app.get('/')
def hello_world():
    return {'message': 'Hello, World'}
    #return render_template("frontend/public/index.html")
@app.post('/prediction')
def prediction(data:disease):
    data=data.dict()
    #pars=[int(x) for x in  request.form.values()]
    
    #pars=[63,1,3,145,233,1,0,150,0,2,0,0,1]
    age=data['age']
    sex=data['sex']
    cp=data['cp']
    trestbps=data['trestbps']
    chol=data['chol']
    fbs=data['fbs']
    restecg=data['restecg']
    thalach=data['thalach']
    exang=data['exang']
    oldpeak=data['oldpeak'] 
    slope=data['slope'] 
    ca=data['ca'] 
    thal=data['thal']
    
    
    
    preds=mod.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    print(preds)
    #print(preds)
    return str(preds)
    #return render_template('heart.html',pred='Chance of getting a heart disease is {}'.format(preds))

if __name__=='__main__':
     uvicorn.run(app,host="127.0.0.1",port=8000)
    
