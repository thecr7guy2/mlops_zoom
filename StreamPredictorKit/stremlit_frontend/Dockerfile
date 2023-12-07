FROM python:3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY frontend.py /app/

COPY images/ /app/images/

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "frontend.py"]

