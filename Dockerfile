FROM tensorflow/tensorflow:latest-py3-jupyter

MAINTAINER lolik samuel

ADD ./requirements.txt /app/

WORKDIR /app

#RUN apt-get update  && apt-get install -y build-essential mpich libpq-dev

#RUN conda install pip

# RUN pip install --upgrade pip \
#  && pip install -r requirements.txt

RUN pip install -r requirements.txt

CMD [ "python", "./rl_dqn.py" ]

