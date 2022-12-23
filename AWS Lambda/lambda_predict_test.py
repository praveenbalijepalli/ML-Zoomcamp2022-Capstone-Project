import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations' # AWS ECR Docker Endpoint/ Lambda Endpoint

url = "https://w6juezx0eg.execute-api.ap-south-1.amazonaws.com/test/predict"   # AWS RestAPI-Lambda Endpoint

data = {'url':"https://raw.githubusercontent.com/praveenbalijepalli/mlzoomcamp-capstone-1/main/sample%20images%20for%20testing/normal.png"} 

response = requests.post(url, json=data).json()

print(f"AWS RestAPI-Lambda prediction service endpoint: {url}")
print(f"Response to {data} sent as a post request and the response: {response}")

 

 
 