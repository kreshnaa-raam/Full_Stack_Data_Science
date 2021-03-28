import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data_loader = load_iris()

X_data = data_loader.data
X_columns = data_loader.feature_names
x = pd.DataFrame(X_data, columns=X_columns)

y_data = data_loader.target
y = pd.Series(y_data, name='target')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

sl_min = np.min(x['sepal length (cm)'].values)
sl_max = np.max(x['sepal length (cm)'].values)

sw_min = np.min(x['sepal width (cm)'].values)
sw_max = np.max(x['sepal width (cm)'].values)

pl_min = np.min(x['petal length (cm)'].values)
pl_max = np.max(x['petal length (cm)'].values)

pw_min = np.min(x['petal width (cm)'].values)
pw_max = np.max(x['petal width (cm)'].values)

print(sl_min, sl_max)
print(sw_min, sw_max)
print(pl_min, pl_max)
print(pw_min, pw_max)