# Import libraries
import numpy as np
import pandas as pd
import category_encoders as cat_enc
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

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
def split_train_data(data):
    y = data['dep_delayed_15min']
    X = data.drop('dep_delayed_15min', axis=1)
    return X, y

X, y = split_train_data(train)

# Apply train test split with 25% data for validation
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25)

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

# Set hyperparameter index
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=3)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 20, num=3)]
# Minimum number of samples required to split a node
min_samples_split = [10, 20]

# Create hyperparameter grid for CV Search for best model
hyper_param_grid = {'classifier__n_estimators': n_estimators,
               'classifier__max_depth': max_depth,
               'classifier__min_samples_split': min_samples_split}

# Set grid search using roc_auc optimization with 3 fold cv
CV = GridSearchCV(classifier_pipe, hyper_param_grid, n_jobs=-1, scoring='roc_auc', verbose=2, cv=3)

# fit the model
CV.fit(X_train, y_train)

# Predict on validation data and generate scores
target_names = y_validation.unique().astype(str)
y_pred = CV.predict(X_validation)
print(classification_report(y_validation, y_pred, target_names=target_names))
print("{}{}".format("Cross - Validation: ", CV.best_score_))
print("{}{}".format("Validation: ", CV.score(X_validation, y_validation)))


print('Reading test df')
test = pd.read_csv(path + "airline_delay_test - airline_delay_test_new.csv")
test = date_features(test)
X, y = split_train_data(test)
print("{}{}".format("Holdout: ", CV.score(X,y)))

pickle.dump(CV, open('model_randomForest_v1.sav', 'wb'))
