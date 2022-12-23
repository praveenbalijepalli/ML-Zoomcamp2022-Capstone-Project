import requests

url = "http://127.0.0.1:9696/predict"

data = {'url':"https://raw.githubusercontent.com/praveenbalijepalli/mlzoomcamp-capstone-1/main/sample%20images%20for%20testing/normal.png"} 

response = requests.post(url, json=data).json()

print(response)

 

 
 