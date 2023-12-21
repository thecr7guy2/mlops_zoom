FROM python:3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv


WORKDIR /app

COPY ["train/Pipfile","train/Pipfile.lock", "./"]


#as docker is already isolated, we skip creating a virtual env
RUN pipenv install --system --deploy

