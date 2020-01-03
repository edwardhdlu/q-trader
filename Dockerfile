#FROM python:3.7-alpine3.10
FROM continuumio/anaconda

ADD ./requirements.txt /code/

WORKDIR /code

#RUN apt-get update \
#    && apt-get install -y build-essential mpich libpq-dev

# should merge to top RUN to avoid extra layers - for debug only :/
RUN pip install -r requirements.txt

#python rl_dqn.py