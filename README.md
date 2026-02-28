# DEWV-STMV-G: River Water Quality Prediction Model

## Overview
DEWV-STMV-G is a machine learning model designed for river water quality prediction. The model combines Graph Neural Networks (GNN), temporal features, meteorological data, and spatial features to predict the concentrations of Total Nitrogen (TN) and Total Phosphorus (TP) at different time horizons (3 hours, 6 hours, 12 hours, 24 hours).

The model includes the following components:
1. **GNN** for capturing spatial relationships between nodes in the river network.
2. **GRU (Gated Recurrent Unit)** for modeling time-series data related to flow and water quality.
3. **MeteoTransformer** for processing meteorological data.
4. **Gate-Injection Mechanism** to adjust the model's predictions based on gate control events.

## Features
- Predicts TN and TP concentrations at multiple future time points.
- Utilizes spatial features (such as flow, land use, and distance decay) and temporal features (such as meteorological data and time-series data).
- Supports training and evaluation workflows.

## Usage

### Model Architecture

#### 1. **Data Preprocessing**

- **StandardScaler**: The model uses standardization to preprocess input data, including flow, TN, and TP concentrations.
- **Time Features**: Time features are created through sine and cosine transformations of hours, days of the week (dow), and days of the year (doy) to capture periodic patterns in time.

#### 2. **Graph Neural Network (GNN)**

- **GCNConv**: The model uses Graph Convolution Networks (GCN) to process the spatial relationships between nodes in the river network.
- **Edge Weights**: Edges in the GNN are dynamically weighted based on flow magnitude and distance decay.

#### 3. **Gated Recurrent Unit (GRU)**

- **GRU**: The model uses GRU (Gated Recurrent Unit) to capture time dependencies related to TN and TP concentrations.

#### 4. **MeteoTransformer**

- **MeteoTransformer**: A transformer model used to process meteorological data, extracting information such as temperature, humidity, and precipitation.

#### 5. **Gate Injection Mechanism**

- The model includes a gate control mechanism, where the model's predictions are adjusted based on gate control events.

### Evaluation Metrics

The model uses the following metrics for evaluation:
- **R²**: The proportion of variance in the target variable that can be predicted by the input features.
- **RMSE**: The root mean square error between the predicted values and actual values.
- **MAE**: The mean absolute error between the predicted and actual values.
- **MAPE**: The mean absolute percentage error, measuring the model's accuracy.

Additionally, the model calculates metrics related to gate prediction, such as F1 score, precision, recall, and PR-AUC.

### Results

After training, the model achieves the following performance (example metrics):
- **R²**: 0.85
- **RMSE**: 0.12
- **MAE**: 0.08
- **MAPE**: 5.2%
