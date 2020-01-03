FROM python:3.7-alpine3.10

MAINTAINER lolik samuel

ADD ./requirements.txt /app/

WORKDIR /app

#RUN apt-get update  && apt-get install -y build-essential mpich libpq-dev

RUN pip install -r requirements.txt

CMD [ "python", "./rl_dqn.py" ]
