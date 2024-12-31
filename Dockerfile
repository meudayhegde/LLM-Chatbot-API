FROM python:3.12.8-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update
RUN apt-get install libreoffice -y

COPY . .

CMD [ "python3", "./app.py" ]
