kaggle competitions download -c ieee-fraud-detection
mkdir -p data
unzip -o ieee-fraud-detection.zip -d data
rm -rf ieee-fraud-detection.zip
rm ./data/sample_submission.csv ./data/test_identity.csv ./data/test_transaction.csv
mv ./data/train_identity.csv ./data/identities.csv
mv ./data/train_transaction.csv ./data/transactions.csv