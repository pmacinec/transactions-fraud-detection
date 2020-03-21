# Transactions Fraud Detection

**Authors:** [Peter Macinec](https://github.com/pmacinec), [Timotej Zatko](https://github.com/timzatko)

## Installation and running

To run this project, please make sure you have Docker installed. After, follow the steps:
1. Get into project root repository.
1. Download data (you need to have [Kaggle API](https://github.com/Kaggle/kaggle-api) installed). Don't for get to accept the [rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) of competition.
    ```
    ./scripts/download_data.sh
    ```
1. Build docker image:
    ```
    ./scripts/build.sh
    ```
1. Run docker container using command: 
    ```
    ./scripts/run.sh
    ```
1. Run initial dataset processing:
    1. Get into docker container:
        ```
        docker exec -it transactions_fraud_detection_con bash    
        ```
   1. Run script with data preprocessing:
        ```
        python src/preprocessing/initial_preprocessing.py
        ``` 
        **Note:** You can run this script also outside the container, but make sure you have python with `pandas` library installed. 
