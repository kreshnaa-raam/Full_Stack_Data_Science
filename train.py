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

# Set hyper parameter index
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=3)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 20, num=3)]
# Minimum number of samples required to split a node
min_samples_split = [10, 20]

# Create hyper parameter grid for CV Search for best model
hyper_param_grid = {'classifier__n_estimators': n_estimators,
                    'classifier__max_depth': max_depth,
                    'classifier__min_samples_split': min_samples_split}

# Set grid search using roc_auc optimization with 3 fold cv
rf_CV = GridSearchCV(classifier_pipe, hyper_param_grid, n_jobs=-1, scoring='roc_auc', verbose=2, cv=3)

# fit the model
rf_CV.fit(X_train, y_train)

# Predict on validation data and generate scores
target_names = y_validation.unique().astype(str)
y_pred = rf_CV.predict(X_validation)
print(classification_report(y_validation, y_pred, target_names=target_names))
print("Cross - Validation: ", rf_CV.best_score_)
print("Validation: ", rf_CV.score(X_validation, y_validation))

# Import test data
test = pd.read_csv(path + "airline_delay_test - airline_delay_test_new.csv")
test = date_features(test)
X, y = split_data(test)

# Check test performance
print("Test Score: ", rf_CV.score(X, y))


# export model as pickle file
pickle.dump(rf_CV, open('model_randomForest_v1.sav', 'wb'))

