FROM public.ecr.aws/lambda/python:3.9

RUN pip install --upgrade pip
RUN pip install pillow  
# RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY cerv_fracture_model.tflite . 

COPY lambda_function.py .

CMD  [ "lambda_function.lambda_handler" ]

