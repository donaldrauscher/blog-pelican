FROM gcr.io/cloud-builders/gcloud

ENV SASS_VERSION 1.3.2
ENV PATH /builder/dart-sass:${PATH}

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade setuptools \
  && pip install --no-cache-dir --upgrade -r requirements.txt

RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget -q -O /builder/dart-sass.tar.gz https://github.com/sass/dart-sass/releases/download/${SASS_VERSION}/dart-sass-${SASS_VERSION}-linux-x64.tar.gz \
  && tar xvzf /builder/dart-sass.tar.gz --directory=/builder \
  && rm /builder/dart-sass.tar.gz

ENTRYPOINT ["pelican"]