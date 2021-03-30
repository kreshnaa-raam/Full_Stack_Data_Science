# Import libraries
import numpy as np
import pandas as pd
import category_encoders as cat_enc
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Import data
path = "/Users/kreshnaa.raam/Documents/DR_Projects/FSDS/"
train = pd.read_csv(path + "airline_delay_train - airline_delay_train_new.csv")


# Function for date based features
def date_features(data):
    data['FlightDate'] = pd.to_datetime(data['FlightDate'])
    data['day'] = pd.DatetimeIndex(data['FlightDate']).day.astype('category')
    data['month'] = pd.DatetimeIndex(data['FlightDate']).month.astype('category')
    data['year'] = pd.DatetimeIndex(data['FlightDate']).year.astype('category')
    data['hour'] = pd.to_datetime(data['DepTime'], format='%H:%M').dt.hour.astype('category')
    data['minutes'] = pd.to_datetime(data['DepTime'], format='%H:%M').dt.minute.astype('category')
    data['DepTime'] = pd.to_datetime(data['DepTime'], format='%H:%M').dt.time
    return data


# Apply FE on train data
train = date_features(train)


# Assign numeric and categorical variable list
numerical_cols = ['Distance']
categorical_cols = ['UniqueCarrier', 'Origin', 'Dest', 'Day_of_Week', 'year', 'month', 'day', 'hour', 'minutes']


# Store predictors and target in two different variables
def split_data(data):
    Y = data['dep_delayed_15min']
    X = data.drop('dep_delayed_15min', axis=1)
    return X, Y


X, Y = split_data(train)

# Apply train test split with 25% data for validation
X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.25)

# Creating ss transformer to scale the continuous numerical data with StandardScaler()
numerical_transformer = Pipeline(steps=[('ss', StandardScaler())])

# Creating categorical transformer using ordinal encoder
categorical_transformer = Pipeline(steps=[('ordinal', cat_enc.OrdinalEncoder())])

# Creating preprocess column transformer to combine the ss and ohe pipelines
preprocess = ColumnTransformer(
    transformers=[
        ('cont', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Creat classifier pipeline using random forest
classifier_pipe = Pipeline(steps=[('preprocessor', preprocess),
                                  ('classifier', RandomForestClassifier())])

param_grid = {'classifier__n_estimators': [400]}


# Set grid search using roc_auc optimization with 3 fold cv
rf_CV = GridSearchCV(classifier_pipe,
                     param_grid = param_grid,
                     n_jobs=-1,
                     scoring='roc_auc',
                     verbose=2,
                     cv=3)

# fit the model
model = rf_CV.fit(X_train, y_train)

#Saving the model with pickle
import pickle
# save the model to disk
model_name  = 'model.pkl'
pickle.dump(model, open(model_name, 'wb'))
print("[INFO]: Finished saving model...")

