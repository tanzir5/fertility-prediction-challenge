"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import time 
from tqdm import tqdm
import pickle
import numpy as np
#import datatable as dt

TARGET = 'new_child'
KEY = 'nomem_encr'

def preprocess_background_data(background_data_path, dtype_mapping):
    # Load the background data
    background_data = pd.read_csv(background_data_path, dtype=dtype_mapping)

    # Convert 'wave' to str if not already, ensuring sorting works as expected
    background_data['wave'] = background_data['wave'].astype(str)

    # Sort by KEY and 'wave' in descending order
    background_data.sort_values(
        by=[KEY, 'wave'], 
        ascending=[True, False], 
        inplace=True
    )

    # Aggregate using the first non-null value in each group
    def first_non_null(series):
        return series.dropna().iloc[0] if not series.dropna().empty else pd.NA

    # Group by KEY
    background_data_latest = background_data.groupby(
        KEY
    ).agg(first_non_null)

    # Reset index to undo the grouping effect
    background_data_latest.reset_index(inplace=True)

    # Drop the 'wave' column
    background_data_latest.drop('wave', axis=1, inplace=True)

    return background_data_latest

def get_dtype_mapping(codebook):
    dtype_mapping = {}
    for _, row in tqdm(codebook.iterrows()):
        if row['type_var'] == 'numeric':
            dtype_mapping[row['var_name']] = 'float32'
        else:# row['type_var'] == 'categorical':
            dtype_mapping[row['var_name']] = 'str'
    return dtype_mapping

def load_data(nrows=None, col_subset=None):
    codebook = pd.read_csv('PreFer_codebook.csv')
    dtype_mapping = get_dtype_mapping(codebook)
    
    return codebook

def merge_data(train_data, train_background):
    train_combined = train_data.merge(
        train_background, on=KEY, how='left'
    )
    return train_combined

def encode_and_clean_data(train_combined, codebook):
    # Classify columns based on their type_var from the codebook
    categorical_vars = (
        codebook[codebook['type_var'] == 'categorical']['var_name']
    )
    categorical_vars = [col for col in categorical_vars if col in train_combined.columns]

    open_ended_vars = (
        codebook[codebook['type_var'] == 'response to open-ended question']
        ['var_name']
    )
    open_ended_vars = [col for col in open_ended_vars if col in train_combined.columns]
    
    print("NOW1")

    character_condition = (
        codebook['type_var'] == 'character [almost exclusively empty strings]'
    )
    date_time_condition = codebook['type_var'] == 'date or time'
    ignore_vars = (
        codebook[character_condition | date_time_condition]['var_name']
    )
    ignore_vars = [col for col in ignore_vars if col in train_combined.columns]
    
    print("NOW2")

    # Drop columns that need to be ignored
    train_combined.drop(ignore_vars, axis=1, inplace=True)

    # st = time.time()
    # # Encode open-ended responses as binary
    for col in tqdm(open_ended_vars):
        train_combined[col] = train_combined[col].notna().astype(int)
    

    # for col in tqdm(categorical_vars):
    #     train_combined[col] = train_combined[col].astype(str)

    # print(f"{time.time()-st} seconds for dtype fixing")
    
    # Which categorical variables to one-hot encode and which to target encode
    max_categories_for_one_hot = 15
    low_cardinality_cats = (
        codebook[
            (codebook['unique_values_n'] <= max_categories_for_one_hot) & 
            (codebook['type_var'] == 'categorical')
        ]['var_name']
    )
    low_cardinality_cats = [col for col in low_cardinality_cats if col in train_combined.columns]

    high_cardinality_cats = (
        codebook[
            (codebook['unique_values_n'] > max_categories_for_one_hot) & 
            (codebook['type_var'] == 'categorical')
        ]['var_name']
    )
    high_cardinality_cats = [col for col in high_cardinality_cats if col in train_combined.columns]

    # Initialize encoders
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    target_encoder = TargetEncoder()
    print("YOLO")
    # Apply one-hot encoding
    if len(low_cardinality_cats) > 0:
        st = time.time()
        one_hot_encoded = pd.DataFrame(
            oh_encoder.fit_transform(
                train_combined[low_cardinality_cats].fillna('Missing')
            ), 
            columns=oh_encoder.get_feature_names_out(low_cardinality_cats)
        )
        print(f"{time.time()-st} seconds for ohe")
        train_combined.drop(low_cardinality_cats, axis=1, inplace=True)
        train_combined = pd.concat([train_combined, one_hot_encoded], axis=1)
        print(f"{time.time()-st} seconds for ohe total")
    print("1YOLO")
    # Apply target encoding
    if len(high_cardinality_cats) > 0:
        st = time.time()
        train_combined[high_cardinality_cats] = target_encoder.fit_transform(
            train_combined[high_cardinality_cats].fillna('Missing'), 
            train_combined[TARGET]
        )  
        print(f"{time.time()-st} seconds for te")
        
    return train_combined

def fill_missing_with_mean(df):
    # Compute the mean for each numeric column
    means = df.drop(columns=[TARGET, KEY]).mean()
    
    # Fill missing values with the computed means for all columns except the 
    # excluded ones
    df.update(df.drop(columns=[TARGET, KEY]).fillna(means))
    return df


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## This script contains a bare minimum working example
    # Create new variable with age
    return df

    train_data = df
    train_background = background_df
    codebook = load_data()
    train_combined = merge_data(train_data, train_background)
    train_combined = encode_and_clean_data(train_combined, codebook)
    # Convert KEY to string
    train_combined[KEY] = train_combined[KEY].astype(str)
    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model

    with open(model_path, 'rb') as f:
      model = pickle.load(f)


    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = np.random.randint(0, 2, len(df))

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
