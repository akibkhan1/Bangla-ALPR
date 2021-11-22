FROM python:3.7
ENV PYTHONBUFFERED=1

USER root
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/