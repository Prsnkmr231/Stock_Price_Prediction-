
# Stock Price Prediction Using LSTM

This project demonstrates how to predict stock prices using Long Short-Term Memory (LSTM) neural networks. The implementation uses TensorFlow and Keras to build, train, and evaluate the model, leveraging historical stock price data.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing and Evaluation](#testing-and-evaluation)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

---

## Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- Pandas
- Scikit-learn
- NumPy
- Matplotlib

---

## Dataset

The dataset used in this project is the historical stock price data of Apple Inc. (AAPL), stored in a CSV file (`AAPL.csv`). The dataset should have the following columns:

- `date`: Date of the record.
- `open`: Opening price.
- `high`: Highest price.
- `low`: Lowest price.
- `close`: Closing price (used for prediction).
- `volume`: Number of shares traded.

---

## Installation

1. Clone the repository or download the source code.
2. Install the required dependencies using pip:
   ```bash
   pip install tensorflow pandas scikit-learn matplotlib numpy
   ```

---

## Preprocessing

### Steps:

1. **Load the Dataset:**
   - The dataset is loaded using Pandas.
   - The `close` column is selected for training.

2. **Normalize Data:**
   - MinMaxScaler is applied to scale the data to the range `[0, 1]`.

3. **Split Data:**
   - The data is split into training (70%) and testing (30%) sets.

4. **Dataset Creation:**
   - A helper function `create_dataset` is defined to generate feature and target pairs for training and testing.

5. **Reshaping Data:**
   - Input data is reshaped to 3D format `[samples, timesteps, features]` for the LSTM model.

---

## Model Architecture

### LSTM Layers:

1. **Input Layer:**
   - Accepts sequences of length 100 with 1 feature.

2. **Hidden Layers:**
   - 3 LSTM layers with 50 units each.
   - The first two LSTM layers return sequences.

3. **Output Layer:**
   - A Dense layer with 1 unit for predicting stock prices.

### Compilation:

- Optimizer: `Adam`
- Loss Function: `Mean Squared Error`

---

## Training

The model is trained for 100 epochs with a batch size of 64. The validation dataset is used to monitor performance during training.

### Output:

- Loss values are printed for both training and validation sets.

---

## Testing and Evaluation

### Steps:

1. **Predict:**
   - Predictions are generated for both training and testing sets.

2. **Inverse Transform:**
   - Predicted values are transformed back to their original scale using MinMaxScaler.

3. **Calculate RMSE:**
   - Root Mean Squared Error (RMSE) is calculated for training and testing predictions.

---

## How to Run

1. Ensure you have the dataset `AAPL.csv` in the appropriate directory.
2. Update the file path in the script if necessary:
   ```python
   df = pd.read_csv("/content/drive/MyDrive/NLP_Datasets/AAPL.csv")
   ```
3. Run the script:
   ```bash
   python stock_price_lstm.py
   ```
4. The model will train, and predictions will be evaluated. RMSE values will be printed.

---

## Conclusion

This project demonstrates the use of LSTM for time series prediction. The model effectively predicts stock prices, and its performance can be further enhanced by tuning hyperparameters or adding more features to the dataset.

