FROM python:3.7
ENV PYTHONBUFFERED=1

RUN apt-get update 

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
COPY ./entrypoint.sh /
ENTRYPOINT ["sh", "/entrypoint.sh"]