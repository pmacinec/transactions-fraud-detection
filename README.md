# Transactions Fraud Detection

**Authors:** Peter Macinec, Timotej Zatko

## Installation and running

To run this project, please make sure you have Docker installed. After, follow the steps:
1. Get into project root repository.
1. Build docker image:
    ```
    docker build -t transactions_fraud_detection .
    ```
1. Run docker container using command: 
    ```
    docker run -it --name transactions_fraud_detection_con --rm -p 8888:8888 -v $(pwd):/project/ transactions_fraud_detection
    ```

