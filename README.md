# Used Bike Price Predictor

## Overview
The Used Bike Price Predictor is a machine learning project designed to predict the price of used bikes using a K-Nearest Neighbors (KNN) model. The project includes data preprocessing, feature engineering, model training, and evaluation.

## Project Structure
- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for data exploration, feature engineering, and model development.
- `src/`: Source code for data preprocessing, feature engineering, model training, and evaluation.
- `models/`: Saved models and model performance metrics.
- `README.md`: This file.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/used-bike-price-predictor.git
    cd used-bike-price-predictor
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data
The dataset contains features such as:
- Bike Age (years)
- Kilometers Driven
- Bike Model
- Engine Capacity (cc)
- Location
- Brand

## Feature Engineering
Feature tuning was performed to improve the model's performance. The selected features include:
- `Bike Age`
- `Kilometers Driven`
- `Engine Capacity`
- `Location`
- `Brand`
- `Bike Model`

## Model
The K-Nearest Neighbors (KNN) model was used for predicting the prices of used bikes. The following steps were taken:
1. Data normalization.
2. Model training with KNN.
3. Cross-validation using `cross_val_score` to improve accuracy.
4. Model evaluation using `r2 square` for accuracy.

## Hyperparameters
The following hyperparameters were tuned:
1. Number of neighbors (k)
2. Weights
3. Algorithm
4. Leaf size
5. Metric
6. P

## Evaluation
The model was evaluated using `r2 square` and achieved an accuracy of 89% with a standard deviation of 3%.

## Results
The final model's performance:
- **Accuracy**: 89%
- **Standard Deviation**: 3%

## Conclusion
The KNN model effectively predicts the prices of used bikes with a high degree of accuracy. The feature tuning and cross-validation techniques significantly contributed to the model's performance.

## Future Work
- Explore other machine learning models to compare performance.
- Incorporate additional features to further improve the model's accuracy.
- Deploy the model as a web application for user-friendly interaction.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

