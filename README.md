# Transactions Fraud Detection

**Authors:** [Peter Macinec](https://github.com/pmacinec), [Timotej Zatko](https://github.com/timzatko)

## Installation and running

To run this project, please make sure you have Docker installed. After, follow the steps:
1. Get into project root repository.
1. Download data, you need to have [Kaggle API](https://github.com/Kaggle/kaggle-api) installed. Don't for get to accept the [rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) of competition.
    ```
    ./download_data.sh
    ```
1. Build docker image:
    ```
    ./build.sh
    ```
1. Run docker container using command: 
    ```
    ./run.sh
    ```

