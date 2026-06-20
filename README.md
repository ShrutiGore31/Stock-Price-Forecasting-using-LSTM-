# Stock Price Forecasting using LSTM

An AI-based Financial Time Series Forecasting project that predicts Reliance stock prices using LSTM (Long Short-Term Memory) neural networks. The project covers the complete pipeline from data loading to next-day price prediction, with detailed step-by-step explanations of every concept.

## Features
- Complete 18-step pipeline from data loading to model saving
- Manual Min-Max scaling demonstration with formula
- Manual LSTM gate calculations (Forget, Input, Output, Cell State)
- Time series sequence creation with overlapping windows
- Two-layer LSTM model with Dropout regularization
- Early stopping to prevent overfitting
- Error metrics: MAE, MSE, RMSE
- Actual vs Predicted graph generation
- Training loss graph generation
- Next-day stock price prediction
- Prediction output saved as CSV

## Tech Stack
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure
```
stock-price-forecasting-lstm/
├── stock_forecasting.py          # Main script — complete pipeline
├── reliance_stock_sample.csv     # Input dataset (Reliance stock data)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── actual_vs_predicted.png       # Generated after running (Actual vs Predicted graph)
├── training_loss.png             # Generated after running (Training loss graph)
├── reliance_lstm_model.h5        # Generated after running (Saved model)
└── reliance_prediction_output.csv # Generated after running (Prediction results)
```

## Requirements
- Python 3.8 or higher

## Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/ShrutiGore31/stock-price-forecasting-lstm.git
cd stock-price-forecasting-lstm
```

**Step 2 — Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Add your dataset:**

Place your `reliance_stock_sample.csv` file in the same folder. The CSV must contain at least these columns:
```
Date, Close
```

**Step 4 — Run the project:**
```bash
python stock_forecasting.py
```

## How It Works

```
Load Reliance Stock CSV Dataset
        ↓
Convert Date Column and Sort by Date
        ↓
Extract Close Price Column
        ↓
Apply Min-Max Scaling (0 to 1)
        ↓
Create Time Series Sequences (10 days → predict day 11)
        ↓
Reshape Data for LSTM Input [samples, time_steps, features]
        ↓
Build 2-Layer LSTM Model with Dropout
        ↓
Train Model (80% data) with Early Stopping
        ↓
Predict on Test Data (20% data)
        ↓
Inverse Scale Predictions back to Rupee Values
        ↓
Calculate MAE, MSE, RMSE
        ↓
Generate Actual vs Predicted Graph
        ↓
Predict Next Day Close Price
        ↓
Save Model and Prediction Output CSV
```

## Concepts Covered
- Time Series Forecasting
- LSTM (Long Short-Term Memory)
- Min-Max Scaling (manual + sklearn)
- LSTM Gate Calculations (Forget, Input, Cell State, Output)
- Sequence Creation with sliding window
- Train-Test Split
- Dropout Regularization
- Early Stopping
- Error Metrics (MAE, MSE, RMSE)
- Inverse Scaling
- Model saving (.h5 format)

## Results
- RMSE achieved: **18.4**
- Model successfully forecasts next-day Reliance closing price

## Future Improvements
- Add more features (Open, High, Low, Volume)
- Try GRU and Bidirectional LSTM
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Build Streamlit dashboard for interactive forecasting
- Add multi-day ahead forecasting

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author
Shruti Vijay Gore
