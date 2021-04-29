import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import os
import io
from typing import List, Optional
from scipy.special import expit
g_code_dir = None

schema = {"UniqueCarrier": "object",
          "Origin": "object",
          "Dest": 'object',
          "Distance": 'int64',
          "Day_of_Week":'object'}


  
def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir

def read_input_data(input_binary_data):
    data = pd.read_csv(io.BytesIO(input_binary_data))
    data.drop(['FlightDate','DepTime'],axis=1,inplace = True)

    #Saving this for later
    return data

def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir = str,
    class_order: Optional[List[str]] = None,
    #row_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> None:

    X.drop(['FlightDate','DepTime'],axis=1,inplace=True)
    X = X.astype(schema)

    #Preprocessing for categorical features
    categorical_features = ['UniqueCarrier', 'Origin', 'Dest']
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    #Preprocessor with all of the steps
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)])

    # Full preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    #Train the model-Pipeline
    pipeline.fit(X,y)

    #Preprocess x
    preprocessed = pipeline.transform(X)

    #I could also train the model with the sparse matrix. I transform it to
    preprocessed = pd.DataFrame.sparse.from_spmatrix(preprocessed)
    
    model = RandomForestClassifier(n_estimators = 5)
   
    model.fit(preprocessed,y)


    joblib.dump(pipeline,'{}/preprocessing.pkl'.format(output_dir))
    joblib.dump(model,'{}/model.pkl'.format(output_dir))


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.
    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.
    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    Returns
    -------
    pd.DataFrame
    """

    #Handle null values in categories and numerics
    for c,dt in schema.items():
        if dt =='object':
            data[c] = data[c].fillna('missing')
        else:
            data[c] = data[c].fillna(0)

    pipeline_path = 'preprocessing.pkl'
    pipeline = joblib.load(os.path.join(g_code_dir, pipeline_path))
    preprocessed = pipeline.transform(data)
    preprocessed = pd.DataFrame.sparse.from_spmatrix(preprocessed)
    
    return preprocessed

def load_model(code_dir):
    model_path = 'model.pkl'
    model = joblib.load(os.path.join(code_dir, model_path))
    return model

def score(data, model, **kwargs):
    results = model.predict_proba(data)
    predictions = pd.DataFrame({'True': results[:, 0], 'False':results[:, 1]})

    return predictions