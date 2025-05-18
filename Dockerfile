FROM python:3.10-slim

WORKDIR /app

COPY . /app
COPY requirements.txt requirements.dev.txt .

RUN pip install --no-cache-dir -r requirements.dev.txt

EXPOSE 5002


ENV FLASK_APP=app.py
ENV FLASK_ENV=production

CMD ["python", "app.py"]

