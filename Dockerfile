FROM python:3.8

COPY ./requirements.txt /app/req.txt

WORKDIR /app

RUN pip install -r req.txt

ADD ./trained_models ./trained_models
#ADD ./static ./static
ADD ./main.py main.py

EXPOSE 5000
CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "main:app" ]