FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt-get install -y chromium-driver

RUN pip install -r requirements.txt

EXPOSE 8080
