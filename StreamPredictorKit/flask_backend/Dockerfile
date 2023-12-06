FROM python:3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock", "./"]

#as docker is already isolated, we skip creating a virtual env
RUN pipenv install --system --deploy


COPY app.py /app/
COPY artifacts/ /app/artifacts/

EXPOSE 6969

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:6969", "app:app"]