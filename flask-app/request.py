import requests

url = 'http://139.59.36.35:80/predict_api'
#url = 'http://0.0.0.0:5000/predict_api'
r = requests.post(url, json={'FlightDate': ["2010-01-16"],
                             'UniqueCarrier': ['9E'],
                             'Origin': ["ATL"],
                             'Dest': ["AUS"],
                             'Distance': ['356']})

print(r.json())

