import requests

url = 'http://0.0.0.0:5000/predict_api'
r = requests.post(url, json={'FlightDate': ["2021-01-27"],
                             'UniqueCarrier': [6],
                             'Origin': ["MIA"],
                             'Dest': ["CLT"],
                             'Distance': [100]})

print(r.json())
