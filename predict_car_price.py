from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pickle
import pandas as pd
import numpy as np

predictor = pickle.load(open('car_price_predictor.pkl', 'rb'))
imputer = pickle.load(open('car_price_imputer.pkl', 'rb'))
scaler = pickle.load(open('car_price_scaler.pkl', 'rb'))
ohe = pickle.load(open('car_price_ohe.pkl', 'rb'))

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

    def predict(self):
        df = pd.DataFrame({
           "name": self.name,
           "year": self.year,
           "selling_price": self.selling_price,
           "km_driven": self.km_driven,
           "fuel": self.fuel,
           "seller_type": self.seller_type,
           "transmission": self.transmission,
           "owner": self.owner,
           "mileage": self.mileage,
           "engine": self.engine,
           "max_power": self.max_power,
           "torque": self. torque,
           "seats": self.seats
        },
        index=[0])
        #реплейсим
        df.mileage = df.mileage.str.replace('kmpl|km/kg', '', regex=True).astype(float)
        df.max_power = pd.to_numeric(df.max_power.str.replace('bhp', '', regex=True), errors='coerce')
        df.engine = df.engine.str.replace('CC', '', regex=True).astype(float)
        df = df.drop(columns=['torque'])
        #пропущенные значения
        df[['mileage', 'max_power', 'seats', 'engine']] = imputer.transform(df[['mileage', 'max_power', 'seats', 'engine']])
        df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)
        #ohe
        cat_features = ['fuel',
                        'seller_type',
                        'transmission', 
                        'owner',
                        'seats'
                        ]
        X_cat_only = ohe.transform(df[cat_features])
        features = [
            'year',
            'km_driven',  
            'mileage',
            'engine',
            'max_power',
            'seats']
        X_numeric = df[features]
        X = np.hstack([X_cat_only, X_numeric])
        X_scaled = scaler.transform(X)
        return np.e ** predictor.predict(X_scaled)[0]


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return item.predict()


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    results = []
    for item in items:
        results.append(item.predict())
    return results