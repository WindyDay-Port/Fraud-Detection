import os
import torch
import joblib
from typing import Dict, Any
from dotenv import load_dotenv
from models.SAE.SparseAutoencoder import SparseAE

def load_all_env_variables():
    load_dotenv()

def load_all_parameters() -> Dict[str, Any]:
    load_dotenv()
    parameters = {}

    # Loading preprocessing parameters
    parameters["power_transformer"] = joblib.load(os.getenv("POWER_TRANSFORMER_PATH"))
    parameters["onehot_encoder"] = joblib.load(os.getenv("ONEHOT_ENCODER_PATH"))
    parameters["ordinal_encoder"] = joblib.load(os.getenv("ORDINAL_ENCODER_PATH"))
    parameters["standard_scaler"] = joblib.load(os.getenv("STANDARD_SCALER_PATH"))
    parameters["minmax_scaler"] = joblib.load(os.getenv("MINMAX_SCALER_PATH"))
    
    # Loading parameters in pickle files
    parameters["freq_billing_city"] = joblib.load(os.getenv("FREQ_BILLING_CITY_PATH"))
    parameters["freq_billing_province"] = joblib.load(os.getenv("FREQ_BILLING_PROVINCE_PATH"))
    parameters["freq_shipping_city"] = joblib.load(os.getenv("FREQ_SHIPPING_CITY_PATH"))
    parameters["freq_shipping_province"] = joblib.load(os.getenv("FREQ_SHIPPING_PROVINCE_PATH"))
    
    # Loading Isolation Forest model
    parameters["isolation_forest"] = joblib.load(os.getenv("ISOLATION_FOREST_PATH"))
    
    # Loading Sparse Autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = SparseAE()
    sae_model.load_state_dict(torch.load(os.getenv("SAE_PATH"), map_location=device))
    sae_model.to(device)
    sae_model.eval()
    parameters["sae_model"] = sae_model
    parameters["sae_device"] = device
    
    return parameters
