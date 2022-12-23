FROM  python:3.9-slim 

RUN pip install --upgrade pip
RUN pip install pipenv
RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl
# RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy    

COPY cerv_fracture_model.tflite .

COPY [ "predict_flask.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve","--listen=0.0.0.0:9696","predict_flask:app"]
 