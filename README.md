# Sentiment Analysis

This repository contains a sentiment analysis model built with PyTorch, trained on a dataset of Amazon product reviews. The model predicts sentiment on a scale from -1 (completely negative) to 1 (completely positive). The dataset used is the [Preprocessed Dataset for Sentiment Analysis](https://www.kaggle.com/datasets/pradeeshprabhakar/preprocessed-dataset-sentiment-analysis) from Kaggle.

## Model Architecture

- **Embedding Layer**: Converts words into dense vectors.
- **Linear Layer**: Projects the averaged embeddings into a single output.
- **Tanh Activation**: Maps the output to a range between -1 and 1 to match the sentiment labels.

The model is trained using mini-batch gradient descent with a batch size of 64 for 1000 iterations.