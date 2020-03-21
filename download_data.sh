rm -rf ieee-fraud-detection.zip
kaggle competitions download -c ieee-fraud-detection
mkdir -p data
unzip -o ieee-fraud-detection.zip -d data
