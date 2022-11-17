# Provides us a working Python 3 environment.
FROM python:3.7-slim-buster

WORKDIR /numerai
# ADD ./modelname/requirements.txt /numerai/requirements.txt
ADD ./destroyai/requirements.txt /numerai/requirements.txt

RUN set -ex \
    && apt-get update \
    && apt-get install --yes --no-install-suggests --no-install-recommends \
        libgomp1 \
    && pip install -r requirements.txt

ADD . /numerai

CMD [ "bash", "entry.sh" ]
