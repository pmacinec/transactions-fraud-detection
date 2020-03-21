#!/usr/bin/env bash

docker run -it --name transactions_fraud_detection_con --rm -p 8888:8888 -v $(pwd):/project/ transactions_fraud_detection