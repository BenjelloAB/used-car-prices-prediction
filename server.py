from flask import Flask, request, jsonify , send_from_directory
import pandas as pd
import numpy as np
import pickle
import json
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

def transform_kilometrage(kilometrage_str):
    s = kilometrage_str.replace(' ', '')
    if 'Plusde' in s:
        return int(s.replace('Plusde', ''))
    else:
        low, high = map(int, s.split('-'))
        return (low + high) / 2

def transform_fiscal_hp(hp_str):
    try:
        return float(''.join(filter(str.isdigit, hp_str)))
    except:
        return None

def normalize_bool_keys(features: dict) -> dict:
    mapping = {
        "electricWindows": "electric_windows",
        "remoteCentralLocking": "remote_central_locking",
        "parkingSensors": "parking_sensors",
        "navigationSystem": "navigation_system/gps",
        "rearViewCamera": "rear_view_camera",
        "speedLimiter": "speed_limiter",
        "alloyWheels": "alloy_wheels",
        "airConditioning": "air_conditioning",
        "cdMp3Bluetooth": "cd/mp3/bluetooth",
        "leatherSeats": "leather_seats",
        "onBoardComputer": "on_board_computer",
        "cruiseControl": "cruise_control",
        "abs": "abs",
        "airbags": "airbags",
        "esp": "esp",
        "sunroof": "sunroof"
    }
    return {mapping.get(k, k): v for k, v in features.items()}

with open(r'./best_model_manual.pkl', 'rb') as f:
    Model = joblib.load(f)

# Load encoders and mappings
with open(r'./brand_encoder.json') as f:
    brand_encoding_map = json.load(f)
with open(r'./model_encoder.json') as f:
    model_encoding_map = json.load(f)
with open(r'./brand_smoothed_encoder.json') as f:
    brand_smoothed_encoding_map = json.load(f)
with open(r'./model_smoothed_encoder.json') as f:
    model_smoothed_encoding_map = json.load(f)
with open(r'./mean_log_price.json') as f:
    mean_log_price = json.load(f)

with open(r'./feature_order.json') as f:
    feature_order = json.load(f)

with open(r'./transmission_target_encoder.pkl', 'rb') as f:
    transmission_te_encoder = joblib.load(f)

scaler = joblib.load(r'./scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')

@app.route('/')
def index():
    # return send_from_directory('./', 'MachineLearningInterface.html')
    return send_from_directory('./', 'testing.html')


@app.route('/predict', methods=['POST'])
def predict():
    voiture = request.get_json()
    print("Reçu :", voiture)
    
    bool_features = normalize_bool_keys(voiture.get("fonctionnalites", {}))

    input_dict = {
        "brand": voiture.get("marque"),
        "model": voiture.get("model"),
        "year": voiture.get("annee"),
        "transmission": voiture.get("transmission"),
        "fuel_type": voiture.get("fuelType"),
        "kilometrage": voiture.get("kilometrage"),
        "origin": voiture.get("origin"),
        "first_owner": voiture.get("firstOwner"),
        "fiscal_horsepower": voiture.get("fiscalHorsepower"),
        "condition": voiture.get("condition"),
        "door_number": voiture.get("doorNumber"),
        **bool_features
    }

    # Transform values
    input_dict["kilometrage_num"] = transform_kilometrage(input_dict["kilometrage"])
    input_dict["fiscal_horsepower_num"] = transform_fiscal_hp(input_dict["fiscal_horsepower"])

    # Target encoding for brand/model
    brand_te = brand_encoding_map.get(input_dict["brand"], mean_log_price)
    model_te = model_encoding_map.get(input_dict["model"], mean_log_price)
    brand_smoothed_te = brand_smoothed_encoding_map.get(input_dict["brand"], mean_log_price)
    model_smoothed_te = model_smoothed_encoding_map.get(input_dict["model"], mean_log_price)

    # Numeric scaling
    numerical_cols = ['door_number', 'fiscal_horsepower_num', 'kilometrage_num', 'year']
    numeric_vals = np.array([[input_dict[col] for col in numerical_cols]])
    numeric_scaled = scaler.transform(numeric_vals)
    scaled_df = pd.DataFrame(numeric_scaled, columns=[f'scaled_{col}' for col in numerical_cols])


    onehot_df = pd.DataFrame([{
    'transmission': input_dict['transmission'],
    'fuel_type': input_dict['fuel_type'],
    'origin': input_dict['origin'],
    'first_owner': input_dict['first_owner'],
    'condition': input_dict['condition'],
    }])
    onehot_encoded = onehot_encoder.transform(onehot_df).toarray()
    onehot_df_final = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out())
    #new : 
    onehot_df_final = onehot_df_final.drop(columns=['transmission_Automatique', 'transmission_Manuelle'], errors='ignore')


    # Boolean features
    bool_df = pd.DataFrame([bool_features])    
    
    #new : Feature engineering
    year_kilometrage_ratio = scaled_df['scaled_year'].iloc[0] / (scaled_df['scaled_kilometrage_num'].iloc[0] + 1e-6)
    model_fiscal_power = model_smoothed_te * scaled_df['scaled_fiscal_horsepower_num'].iloc[0]
    
    
    #new: Target encoding for transmission
    transmission_te = transmission_te_encoder.transform(pd.DataFrame({'transmission': [input_dict["transmission"]]})).iloc[0, 0]

    # Combine all encoded features
    encoded_input = pd.concat([
        scaled_df.reset_index(drop=True),
        onehot_df_final.reset_index(drop=True),
        bool_df.reset_index(drop=True),
        pd.DataFrame({
            'brand_kfold5_te': [brand_te],
            'model_kfold5_te': [model_te],
            'brand_smoothed_te': [brand_smoothed_te],
            'model_smoothed_te': [model_smoothed_te],
            'transmission_te': [transmission_te],
            'year_kilometrage_ratio': [year_kilometrage_ratio],
            'model_fiscal_power': [model_fiscal_power]
        })
    ], axis=1)
    
    model = joblib.load('best_model_manual.pkl')
    with open('./feature_order.json') as f:
        feature_order = json.load(f)
    print("Colonnes dans encoded_input :", encoded_input.columns.tolist())
    print("Colonnes attendues (feature_order) :", feature_order)
    encoded_input = encoded_input[feature_order]

    predicted_log_price = Model.predict(encoded_input)[0]

    predicted_price = np.exp(predicted_log_price)


    print("predicted price : ", predicted_price)
    rounded_price = np.round(predicted_price, 2)  # or round(predicted_price) for no decimals

    print("rounded predicted price : ", rounded_price)

    return jsonify({'prediction': rounded_price})

    # return jsonify({'prediction': predicted_price.tolist()})

if __name__ == '__main__':

    app.run(port=8000, debug=True)