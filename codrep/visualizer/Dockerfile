FROM node:8.16.0-buster-slim AS react-build

RUN mkdir /visualizer

WORKDIR /visualizer

ADD package.json .
ADD package-lock.json .

RUN npm ci && npm audit fix

ADD tsconfig.json .
ADD src src
ADD public public

RUN npm run build

FROM python:3.7.4-alpine3.10

RUN mkdir -p /codrep/visualizer

WORKDIR /codrep/visualizer

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD serve.py .

COPY --from=react-build /visualizer/build build

ENTRYPOINT python serve.py \
    --data /codrep/data \
    --results /codrep/results
