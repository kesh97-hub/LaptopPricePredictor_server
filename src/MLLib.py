import pickle
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor


# convert into categorical
def convert_categorical_data(data):
    with open('src/Resources/category_info.pkl', 'rb') as file:
        category_mapping = pickle.load(file)

    categorical_columns = ['brand', 'model', 'cpu', 'OS', 'special_features', 'graphics', 'graphics_coprocessor']

    for col in categorical_columns:
        category_mapping[col + "_categories"].sort()
        # print(category_mapping[col+"_categories"])
        data[col] = pd.Categorical(data[col], categories=category_mapping[col + '_categories'])
        data[col] = data[col].cat.codes

    return data


def normalize_data(data, type="predictPrice"):
    if type == "predictPrice":
        with open('src/Resources/standardScaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    elif type == 'similarLaptops':
        with open('src/Resources/knnStandardScaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    else:
        return

    data = scaler.transform(data)
    return data


def getTrainingData():
    training_data = pd.read_csv('src/Resources/amazon_laptop_prices_v02_cleaned.csv')
    ordered_columns = ['brand', 'model', 'screen_size', 'cpu', 'ram', 'OS', 'special_features', 'graphics',
                       'graphics_coprocessor', 'harddisk_numeric', 'price']

    training_data = training_data[ordered_columns]
    return training_data


# predict price and return price
def predict_price(data):
    laptop_config = data
    laptop_config_df = pd.DataFrame(data, index=[0])
    laptop_config_df = laptop_config_df.fillna("")
    ordered_columns = ['brand', 'model', 'screen_size', 'cpu', 'ram', 'OS', 'special_features', 'graphics',
                       'graphics_coprocessor', 'harddisk_numeric']

    laptop_config_df = laptop_config_df[ordered_columns]

    # convert columns to categorical codes
    laptop_config_df = convert_categorical_data(laptop_config_df)

    # normalize data
    laptop_config_df = normalize_data(laptop_config_df, type="predictPrice")

    # load model
    with open('src/Resources/xgboostModel.pkl', 'rb') as xgboostModel:
        xgboostModel = pickle.load(xgboostModel)

    predicted_price = xgboostModel.predict(laptop_config_df)
    # print(f'Predicted price: {predicted_price.tolist()[0]}')
    laptop_config['price'] = predicted_price.tolist()[0]

    similar_laptops = getSimilarProducts(laptop_config, 3)

    similar_laptops.append(laptop_config)

    return similar_laptops


def getSimilarProducts(data, n=3):
    with open('src/Resources/knnModel.pkl', 'rb') as knnModel:
        knnModel = pickle.load(knnModel)

    laptop_config_df = pd.DataFrame(data, index=[0])
    laptop_config_df = laptop_config_df.fillna("")

    laptop_config_df = convert_categorical_data(laptop_config_df)

    laptop_config_df = normalize_data(laptop_config_df, type="similarLaptops")

    distances, indices = knnModel.kneighbors(laptop_config_df)
    # print('Indices:', indices)

    training_data = getTrainingData()
    similar_laptops = training_data.iloc[indices.ravel()]
    similar_laptops = similar_laptops.fillna("")
    return similar_laptops.to_dict(orient='records')


def get_brand_price_chart_data():
    training_data = getTrainingData()

    result = training_data.groupby('brand')['price'].mean()

    return result.to_dict()
