import json
import requests

data = {
    "sepal length (cm)": 6.1,
    "sepal width (cm)": 3.2,
    "petal length (cm)": 3.95,
    "petal width (cm)": 1.3
}

ip_address = 'http://64.227.72.137/predict'

r = requests.post(ip_address, json=data)

print(r.text)
print(r.status_code)