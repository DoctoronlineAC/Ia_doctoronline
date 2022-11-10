FROM python:3.8.4-slim-buster 

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT uvicorn --host 0.0.0.0 main:app --reload
  