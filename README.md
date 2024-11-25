# Implement-RNNs-for-Named-Entity-Recognition

This project implements a Recurrent Neural Network (RNN) model for Named Entity Recognition (NER) using PyTorch and explores hyperparameter tuning for optimal performance.

## Project Description

The code trains an RNN-based NER model on the CoNLL-2003 dataset, following these key steps:

1. **Data Preparation:**
   - Loads the CoNLL-2003 dataset (train.txt and test.txt).
   - Preprocesses the data, including building vocabulary and encoding sentences and tags.

2. **Model Implementation:**
   - Defines an RNN model using PyTorch's LSTM layer, embedding layer, and a fully connected layer for classification.

3. **Training and Evaluation:**
   - Trains the model using the Adam optimizer and CrossEntropyLoss.
   - Evaluates the model's performance on a held-out test set.
   - Reports standard NER metrics like precision, recall, and F1-score.

4. **Hyperparameter Tuning:**
   - Explores the impact of various hyperparameters on model performance:
     - **Batch Size:** Experimenting with different batch sizes during training.
     - **Optimizer:** Comparing the performance of different optimizers (e.g., Adam, SGD).
     - **Learning Rate:** Investigating the effect of learning rate on convergence and accuracy.
     - **... (Other relevant hyperparameters)**: Include any other hyperparameters you tuned.

## Requirements

- Python 3.6 or higher
- PyTorch 1.7 or higher
- NumPy
- scikit-learn
- tqdm

## Usage

1. **Data:**
   - Download the CoNLL-2003 dataset and place the `train.txt` and `test.txt` files in the same directory as the code.
2. **Training and Evaluation:**
   - Run the Python script to train and evaluate the model.
3. **Hyperparameter Tuning:**
   - Modify the hyperparameter values in the code to experiment with different settings.
   - Observe the impact on evaluation metrics to identify optimal configurations.

## Results and Observations

## Results and Observations

- **Classification Report:**

| Entity        | Precision | Recall | F1-Score | Support |
| ------------- | --------- | -------- | -------- | ------- |
| O            | 0.97      | 0.96    | 0.96    | 38378  |
| B-ORG        | 0.67      | 0.59    | 0.63    | 1658   |
| B-MISC        | 0.73      | 0.62    | 0.67    | 701    |
| B-PER        | 0.84      | 0.59    | 0.70    | 1580   |
| I-PER        | 0.87      | 0.50    | 0.64    | 1111   |
| B-LOC        | 0.83      | 0.79    | 0.81    | 1656   |
| I-ORG        | 0.28      | 0.79    | 0.41    | 827    |
| I-MISC        | 0.65      | 0.52    | 0.58    | 216    |
| I-LOC        | 0.70      | 0.65    | 0.68    | 255    |
| accuracy     |           |         | 0.91    | 46382  |
| macro avg    | 0.73      | 0.67    | 0.67    | 46382  |
| weighted avg | 0.93      | 0.91    | 0.91    | 46382  |
