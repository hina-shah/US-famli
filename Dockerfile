FROM tensorflow/tensorflow:1.15.0-py3-jupyter


WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y tesseract-ocr && apt-get clean

COPY . .
