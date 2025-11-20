import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader

def load_all_env_variables():
  load_dotenv()

def mse_computing(dataframe: pd.DataFrame, model, device) -> pd.DataFrame:
  dataframe_tensor = torch.tensor(dataframe.values, dtype=torch.float32)
  dataframe_dataloader = DataLoader(dataframe_tensor, batch_size=128, shuffle=False) 

  mse_rates = []
  
  with torch.no_grad():
    for batch in dataframe_dataloader:
      batch_input = batch.to(device)
      _, decoded = model(batch_input)
      error = torch.mean((decoded - batch_input)**2, dim=1)
      mse_rates.extend(error.cpu().numpy())
  
  all_errors = np.array(mse_rates)
  
  dataframe["MSE_rate"] = all_errors
  return dataframe
