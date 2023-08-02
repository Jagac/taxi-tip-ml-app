import requests

from tagiftip.main import load_data

df = load_data()
df = df.drop("tipped", axis=1)
data = df.to_dict(orient="records")

for i in range(len(data)):
    payload = data[i]
    prediction = requests.post("http://127.0.0.1:8000/predict", json=payload)
    print(prediction.json())
