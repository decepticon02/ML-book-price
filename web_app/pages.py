import os
import pickle
from flask import Blueprint, render_template,request
import pandas as pd
import psycopg2
import sys

from sklearn.discriminant_analysis import StandardScaler
sys.path.append("/home/danilovic/PSZ/ml")
from ml.MyLinearRegression import My_Linear_Regression

from ml.db_querries import DbClass

bp = Blueprint("pages", __name__)


@bp.route("/")
def home():
    return render_template("pages/home.html")

@bp.route("/logisticonevsrest",methods=["GET", "POST"])
def logisticonevsrest():
    db = DbClass()
    db.connect()
    kategorije = db.getKategorije()
    izdavaci = db.getIzdavaci()
    godine = db.getGodine()
    povezi = db.getPovez()
    db.close()
    if request.method == "POST":
        with open("web_app/models/onevsrest_reg_model.pkl", 'rb') as file: 
            model,scaler,feature_names = pickle.load(file)

        kategorija = request.form.get('kategorija')
        izdavac = request.form.get('izdavac')
        godina = request.form.get('godina')

        povez = request.form.get('povez')
        strane = request.form.get('strane',type=float)
        format = request.form.get('format',type=float)
        print(kategorija,godina,povez,strane, format)
        
        dict = {'povez': povez, 'godina': godina, 'kategorija': kategorija,'strana':strane,'format':format,'izdavac':izdavac} 
        
        
        x_predict = pd.DataFrame([dict])
        print(x_predict)
        
        x_predict = pd.get_dummies(x_predict, columns=['povez','godina','kategorija','izdavac']) # jos i format strana 

        print(x_predict)
        x_predict_filled = pd.DataFrame(columns=feature_names)

        for column in x_predict.columns:
            if column in x_predict_filled.columns:
                x_predict_filled[column] = x_predict[column]

        x_predict = x_predict_filled.fillna(0)

        print(x_predict)

        x_predict_scaled = scaler.transform(x_predict)
        prediction = model.predict(x_predict_scaled)
        print(prediction)

        return render_template("pages/logistic-one-vs-rest.html", kategorije=kategorije, izdavaci=izdavaci, godine=godine,
                               povez=povezi, result=prediction[0], selected_kategorija=kategorija,selected_izdavac=izdavac,
                               selected_godina=godina, selected_povez=povez, selected_strane=strane,
                               selected_format=format)


    else:
        return render_template("pages/logistic-one-vs-rest.html",kategorije=kategorije,izdavaci=izdavaci,godine=godine,povez=povezi,result=0)


@bp.route("/logisticonevsone",methods=["GET", "POST"])
def logisticonevsone():
    db = DbClass()
    db.connect()
    kategorije = db.getKategorije()
    izdavaci = db.getIzdavaci()
    godine = db.getGodine()
    povezi = db.getPovez()
    db.close()
    if request.method == "POST":
        with open("web_app/models/onevso_reg_model.pkl", 'rb') as file: 
            model,scaler,feature_names = pickle.load(file)

        kategorija = request.form.get('kategorija')
        izdavac = request.form.get('izdavac')
        godina = request.form.get('godina')

        povez = request.form.get('povez')
        strane = request.form.get('strane',type=float)
        format = request.form.get('format',type=float)
        print(kategorija,godina,povez,strane, format)
        
        dict = {'povez': povez, 'godina': godina, 'kategorija': kategorija,'strana':strane,'format':format,'izdavac':izdavac} 
        
        
        x_predict = pd.DataFrame([dict])
        print(x_predict)
        
        x_predict = pd.get_dummies(x_predict, columns=['povez','godina','kategorija','izdavac']) # jos i format strana 

        print(x_predict)
        x_predict_filled = pd.DataFrame(columns=feature_names)

        for column in x_predict.columns:
            if column in x_predict_filled.columns:
                x_predict_filled[column] = x_predict[column]

        x_predict = x_predict_filled.fillna(0)

        print(x_predict)

        x_predict_scaled = scaler.transform(x_predict)
        prediction = model.predict(x_predict_scaled)
        print(prediction)

        return render_template("pages/logistic-one-vs-one.html", kategorije=kategorije, izdavaci=izdavaci, godine=godine,
                               povez=povezi, result=prediction[0], selected_kategorija=kategorija,selected_izdavac=izdavac,
                               selected_godina=godina, selected_povez=povez, selected_strane=strane,
                               selected_format=format)


    else:
        return render_template("pages/logistic-one-vs-one.html",kategorije=kategorije,izdavaci=izdavaci,godine=godine,povez=povezi,result=0)

@bp.route("/logisticmultinomial",methods=["GET", "POST"])
def logisticmultinomial():
    db = DbClass()
    db.connect()
    kategorije = db.getKategorije()
    izdavaci = db.getIzdavaci()
    godine = db.getGodine()
    povezi = db.getPovez()
    db.close()
    if request.method == "POST":
        with open("web_app/models/onevsrest_reg_model.pkl", 'rb') as file: 
            model,scaler,feature_names = pickle.load(file)

        kategorija = request.form.get('kategorija')
        izdavac = request.form.get('izdavac')
        godina = request.form.get('godina')

        povez = request.form.get('povez')
        strane = request.form.get('strane',type=float)
        format = request.form.get('format',type=float)
        print(kategorija,godina,povez,strane, format)
        
        dict = {'povez': povez, 'godina': godina, 'kategorija': kategorija,'strana':strane,'format':format,'izdavac':izdavac} 
        
        
        x_predict = pd.DataFrame([dict])
        print(x_predict)
        
        x_predict = pd.get_dummies(x_predict, columns=['povez','godina','kategorija','izdavac']) # jos i format strana 

        print(x_predict)
        x_predict_filled = pd.DataFrame(columns=feature_names)

        for column in x_predict.columns:
            if column in x_predict_filled.columns:
                x_predict_filled[column] = x_predict[column]

        x_predict = x_predict_filled.fillna(0)

        print(x_predict)

        x_predict_scaled = scaler.transform(x_predict)
        prediction = model.predict(x_predict_scaled)
        print(prediction)

        return render_template("pages/logistic-one-vs-rest.html", kategorije=kategorije, izdavaci=izdavaci, godine=godine,
                               povez=povezi, result=prediction[0], selected_kategorija=kategorija,selected_izdavac=izdavac,
                               selected_godina=godina, selected_povez=povez, selected_strane=strane,
                               selected_format=format)


    else:
        return render_template("pages/logistic-one-vs-rest.html",kategorije=kategorije,izdavaci=izdavaci,godine=godine,povez=povezi,result=0)


@bp.route("/linear",methods=["GET", "POST"])
def linear():
    db = DbClass()
    db.connect()
    kategorije = db.getKategorije()
    izdavaci = db.getIzdavaci()
    godine = db.getGodine()
    povezi = db.getPovez()
    db.close()
    if request.method == "POST":
        with open("web_app/models/linear_reg_model.pkl", 'rb') as file: 
            model,scaler,feature_names = pickle.load(file)

        kategorija = request.form.get('kategorija')
        izdavac = request.form.get('izdavac')
        godina = request.form.get('godina')

        povez = request.form.get('povez')
        strane = request.form.get('strane',type=float)
        format = request.form.get('format',type=float)
        print(kategorija,godina,povez,strane, format)
        
        dict = {'povez': povez, 'godina': godina, 'kategorija': kategorija,'strana':strane,'format':format,'izdavac':izdavac} 
        
        
        x_predict = pd.DataFrame([dict])
        print(x_predict)
        
        x_predict = pd.get_dummies(x_predict, columns=['povez','godina','kategorija','izdavac']) # jos i format strana 

        print(x_predict)
        x_predict_filled = pd.DataFrame(columns=feature_names)

        for column in x_predict.columns:
            if column in x_predict_filled.columns:
                x_predict_filled[column] = x_predict[column]

        x_predict = x_predict_filled.fillna(0)

        print(x_predict)

        x_predict_scaled = scaler.transform(x_predict)
        prediction = model.predict(x_predict_scaled)
        print(prediction)

        return render_template("pages/linear.html", kategorije=kategorije, izdavaci=izdavaci, godine=godine,
                               povez=povezi, result=prediction[0], selected_kategorija=kategorija,selected_izdavac=izdavac,
                               selected_godina=godina, selected_povez=povez, selected_strane=strane,
                               selected_format=format)


    else:
        return render_template("pages/linear.html",kategorije=kategorije,izdavaci=izdavaci,godine=godine,povez=povezi,result=0)

@bp.route("/kmeans")
def kmeans():
    return render_template("pages/kmeans.html")