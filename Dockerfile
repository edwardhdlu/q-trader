FROM python:3.7-alpine3.10

ADD ./requirements.txt /code/

WORKDIR /code

#RUN apt-get update \
#    && apt-get install -y build-essential mpich libpq-dev

# should merge to top RUN to avoid extra layers - for debug only :/
RUN pip3 install -r requirements.txt

#python rl_dqn.py