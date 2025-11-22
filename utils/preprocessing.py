import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    OrdinalEncoder,
    MinMaxScaler,
)
from sklearn.compose import ColumnTransformer

def initial_transformation(main_dataframe: pd.DataFrame) -> pd.DataFrame:
    # Copying the main DataFrame
    dataframe_copy = main_dataframe.copy()
    
    # Dropping unnecessary columns
    dataframe_copy = dataframe_copy.drop(columns='Unnamed: 0', errors='ignore')
    dataframe_copy = dataframe_copy.drop(columns='device_used.1', errors='ignore')
    dataframe_copy = dataframe_copy.drop(columns='city', errors='ignore')
    dataframe_copy = dataframe_copy.drop(columns='province', errors='ignore')
    dataframe_copy = dataframe_copy.drop(columns='country', errors='ignore')

    # Categorizing 'customer_age'
    bins = [float('-inf'), 0, 17, 30, 50, 100]
    labels = ['unspecified', 'under_age', 'young_adult', 'adult', 'senior']
    dataframe_copy['age_category'] = pd.cut(dataframe_copy['age'], 
                                            bins=bins, labels=labels, 
                                            right=True, ordered=True)
    
    # Dropping values which don't adhere to business rules
    dataframe_copy.drop(dataframe_copy[dataframe_copy['age_category'] == 'unspecified'].index, inplace=True)
    dataframe_copy.drop(dataframe_copy[dataframe_copy['age_category'] == 'under_age'].index, inplace=True)
    
    # Categorizing 'account_age'
    account_bins = [0, 30, 180, float('inf')]
    account_labels = ['new', 'established', 'long_term']
    dataframe_copy['account_age_category'] = pd.cut(
        main_dataframe['account_age'], 
        bins=account_bins, 
        labels=account_labels, 
        right=True
    )
    
    # Converting data types
    dataframe_copy['product_category'] = dataframe_copy['product_category'].astype('category')
    dataframe_copy['product_price'] = dataframe_copy['product_price'].astype('float64')
    dataframe_copy['ordered_quantity'] = dataframe_copy['ordered_quantity'].astype('int32')
    dataframe_copy['device_used'] = dataframe_copy['device_used'].astype('category')
    dataframe_copy['payment_method'] = dataframe_copy['payment_method'].astype('category')
    dataframe_copy['payment_amount'] = dataframe_copy['payment_amount'].astype('float64')
    dataframe_copy['payment_hour'] = dataframe_copy['payment_hour'].astype('int32')
    dataframe_copy['timestamp'] = dataframe_copy['timestamp'].astype('datetime64[ns]')
    
    # Categorizing 'payment_hour'
    hour_bins = [-1, 6, 12, 18, 24]
    hour_labels = ['late_night', 'morning', 'afternoon', 'evening']
    dataframe_copy['payment_hour_category'] = pd.cut(dataframe_copy['payment_hour'], 
                                                     bins=hour_bins, labels=hour_labels, 
                                                     right=False)    
    del main_dataframe    
    gc.collect()
    return dataframe_copy

def address_splitting(dataframe: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
    def splitting(address: str):
        parts = address.split(', ') if isinstance(address, str) else []
        street = city = province = country = None

        # In case three symbols or more
        if len(parts) >= 3:
            street = parts[-4]
            city = parts[-3]
            province = parts[-2]
            country = parts[-1]
        
        # In case two symbols
        elif len(parts) == 2:
            city = parts[-2]
            province = parts[-1]
            country = None
        
        # In case one symbol or lesser
        else:
            city = province = country = None

        return pd.Series([street, city, province, country])
    
    new_cols = [f'{prefix}_street', f'{prefix}_city', f'{prefix}_province', f'{prefix}_country']
    dataframe[new_cols] = dataframe[column_name].apply(splitting)    
    gc.collect()
    return dataframe

def processing_sort_timestamp(dataframe: pd.DataFrame) -> pd.DataFrame:    
    dataframe['second_of_minute'] = dataframe['timestamp'].dt.second
    dataframe['minute_of_hour'] = dataframe['timestamp'].dt.minute
    dataframe['day_of_week'] = dataframe['timestamp'].dt.dayofweek
    dataframe['month_of_year'] = dataframe['timestamp'].dt.month
    
    dataframe = dataframe.sort_values(by='timestamp').reset_index(drop=True)
    
    dataframe['is_weekend'] = 1
    dataframe['is_weekend'] = dataframe['is_weekend'].where((dataframe['day_of_week'] == 5) | 
                                                            (dataframe['day_of_week'] == 6), 0)
    
    dataframe['second_sin'] = np.sin(2 * np.pi * dataframe['second_of_minute'] / 60)
    dataframe['second_cos'] = np.cos(2 * np.pi * dataframe['second_of_minute'] / 60)
    
    dataframe['minute_sin'] = np.sin(2 * np.pi * dataframe['minute_of_hour'] / 60)
    dataframe['minute_cos'] = np.cos(2 * np.pi * dataframe['minute_of_hour'] / 60)
    
    dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['payment_hour'] / 24)
    dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['payment_hour'] / 24)
    
    dataframe['day_sin'] = np.sin(2 * np.pi * dataframe['day_of_week'] / 7)
    dataframe['day_cos'] = np.cos(2 * np.pi * dataframe['day_of_week'] / 7)
    
    dataframe['month_sin'] = np.sin(2 * np.pi * dataframe['month_of_year'] / 12)
    dataframe['month_cos'] = np.cos(2 * np.pi * dataframe['month_of_year'] / 12)    
    
    gc.collect()
    return dataframe

def power_transformation(dataframe: pd.DataFrame, encoder: PowerTransformer) -> pd.DataFrame:
    features_to_transform = ['product_price', 'payment_amount']
    dataframe.loc[:, features_to_transform] = encoder.transform(dataframe.loc[:, features_to_transform])
    return dataframe

def one_hot_encoding(dataframe: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    cols_to_encode = ['payment_method', 'device_used', 'product_category', 'billing_country', 'shipping_country']
    
    encoded = encoder.transform(dataframe[cols_to_encode])
    dataframe = pd.concat([dataframe.drop(columns=cols_to_encode), encoded], axis=1)
    return dataframe

def frequency_encoding(dataframe: pd.DataFrame, billing_city: dict, 
                       shipping_city: dict, billing_province: dict, 
                       shipping_province: dict) -> pd.DataFrame:
    dataframe['encoded_billing_province'] = dataframe['billing_province'].map(billing_province)
    dataframe['encoded_shipping_province'] = dataframe['shipping_province'].map(shipping_province)    
    dataframe['encoded_billing_city'] = dataframe['billing_city'].map(billing_city)
    dataframe['encoded_shipping_city'] = dataframe['shipping_city'].map(shipping_city)
    
    dataframe['encoded_billing_province'] = dataframe['encoded_billing_province'].fillna(0)
    dataframe['encoded_shipping_province'] = dataframe['encoded_shipping_province'].fillna(0)
    dataframe['encoded_billing_city'] = dataframe['encoded_billing_city'].fillna(0)
    dataframe['encoded_shipping_city'] = dataframe['encoded_shipping_city'].fillna(0)
    return dataframe

def ordinal_encoding(dataframe: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame:
    ordinal_features = ['age_category', 'account_age_category', 'payment_hour_category']
    dataframe[ordinal_features] = encoder.transform(dataframe[ordinal_features])
    return dataframe

def city_province_is_different(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['street_is_different'] = (
        (dataframe['shipping_street'] != dataframe['billing_street']).astype(int)
    )

    dataframe['province_is_different'] = (
        (dataframe['shipping_province'] != dataframe['billing_province']).astype(int)
    )
    return dataframe

def dropping_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    features_drop = [
    'customerID', 'first_name', 'middle_name', 'last_name', 
    'ip_address', 'billing_address', 'billing_city',
    'billing_province', 'shipping_address', 'shipping_city',
    'shipping_province', 'payment_hour', 'day_of_week',
    'month_of_year', 'timestamp', 'shipping_street',
    'billing_street', 'second_of_minute', 'minute_of_hour'
    ]
    
    dataframe = dataframe.drop(features_drop, axis=1)
    return dataframe

def standard_scaler(dataframe: pd.DataFrame, preprocessor_pipeline: ColumnTransformer) -> pd.DataFrame:
    scaled_array = preprocessor_pipeline.transform(dataframe)    
    output_column_names = preprocessor_pipeline.get_feature_names_out()    
    scaled_dataframe = pd.DataFrame(scaled_array, columns=output_column_names)    
    return scaled_dataframe

def minmax_scaler(dataframe: pd.DataFrame, encoder: MinMaxScaler) -> pd.DataFrame:
    column_names = dataframe.columns
    scaled = encoder.transform(dataframe)
    dataframe = pd.DataFrame(scaled, columns=column_names)
    return dataframe

def master_preprocessing(dataframe: pd.DataFrame, column_name_1: str, prefix_1: str, 
                         column_name_2: str, prefix_2: str, power_encoder: PowerTransformer,  
                         onehot_encoder: OneHotEncoder, freq_billing_city: dict,
                         freq_shipping_city: dict, freq_billing_province: dict,
                         freq_shipping_province: dict, ordinal_encoder: OrdinalEncoder,
                         std_scaler: ColumnTransformer, mm_scaler: MinMaxScaler) -> pd.DataFrame:
    dataframe = initial_transformation(dataframe)
    dataframe = address_splitting(dataframe, column_name_1, prefix_1)
    dataframe = address_splitting(dataframe, column_name_2, prefix_2)
    dataframe = processing_sort_timestamp(dataframe)
    dataframe = power_transformation(dataframe, power_encoder)
    dataframe = one_hot_encoding(dataframe, onehot_encoder)
    dataframe = frequency_encoding(
        dataframe,
        freq_billing_city,
        freq_shipping_city,
        freq_billing_province,
        freq_shipping_province,
    )
    dataframe = ordinal_encoding(dataframe, ordinal_encoder)
    dataframe = city_province_is_different(dataframe)
    dataframe = dropping_features(dataframe)
    dataframe = standard_scaler(dataframe, std_scaler)
    dataframe = minmax_scaler(dataframe, mm_scaler)
    return dataframe
