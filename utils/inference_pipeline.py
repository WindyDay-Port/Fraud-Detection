import os
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from utils.mse_extractor import mse_computing
from utils.param_loader import load_all_parameters
from utils.preprocessing import master_preprocessing
  
def pipeline(data):
  load_dotenv()
  
  dataframe = pd.DataFrame([data])
  
  components = load_all_parameters()
  
  processed_dataframe = master_preprocessing(
    dataframe=dataframe, 
    column_name_1="billing_address",
    prefix_1="billing", 
    column_name_2="shipping_address",
    prefix_2="shipping", 
    power_encoder=components["power_transformer"],
    onehot_encoder=components["onehot_encoder"], 
    freq_billing_city=components["freq_billing_city"],
    freq_shipping_city=components["freq_shipping_city"],
    freq_billing_province=components["freq_billing_province"],
    freq_shipping_province=components["freq_shipping_province"],
    ordinal_encoder=components["ordinal_encoder"], 
    std_scaler=components["standard_scaler"],
    mm_scaler=components["minmax_scaler"],
    )
  
  processed_dataframe = mse_computing(
    processed_dataframe,
    model=components["sae_model"],
    device=components["sae_device"],
  )
  
  isolation_forest = components["isolation_forest"]
  forest_label = isolation_forest.predict(processed_dataframe)
  processed_dataframe["anomaly_score"] = isolation_forest.decision_function(processed_dataframe)
  
  
  xgboost_model = xgb.XGBClassifier()
  xgboost_model.load_model(os.getenv("XGBOOST_PATH"))
  
  final_predictions = xgboost_model.predict(processed_dataframe)
  prediction_proba = xgboost_model.predict_proba(processed_dataframe)[:, 1]
  
  return {
    "fraud_probability": float(prediction_proba[0]),
    "mse_rate": float(processed_dataframe["MSE_rate"].iloc[0]),
    "anomaly_score": float(processed_dataframe["anomaly_score"].iloc[0]),
    "forest_prediction": int(forest_label[0]),
    "final_prediction": int(final_predictions[0]),
  }
