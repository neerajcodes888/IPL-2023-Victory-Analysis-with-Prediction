from flask import Flask, render_template, request
import joblib
import pandas as pd

model = joblib.load('IPL_Prediction_Model.pkl')

app = Flask(__name__)

