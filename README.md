# Used Car Price Prediction – Advanced Regression

## Project Overview

This project focuses on predicting the selling prices of used cars using machine learning regression techniques. The objective is to develop predictive models that estimate vehicle prices based on different characteristics, including vehicle model, age, mileage, engine power, fuel type, transmission type, and other vehicle specifications.

The project applies **Linear Regression, Ridge Regression, and Lasso Regression** to understand the impact of regularisation on predictive performance, coefficient stability, and feature selection.

## Business Problem

Used-car pricing depends on several factors, such as vehicle age, mileage, engine specifications, brand, body type, and vehicle condition. Estimating an appropriate selling price can be challenging for both buyers and sellers.

The objective of this project is to develop a data-driven regression model that can:

- Predict used-car selling prices.
- Identify important factors influencing vehicle prices.
- Understand the effect of regularisation on regression coefficients.
- Compare Linear, Ridge, and Lasso regression models.
- Support more informed and consistent vehicle pricing decisions.

## Dataset Description

The project uses a used-car dataset containing vehicle listings from the AutoScout24 marketplace.

**Original dataset:**
- Records: 15,915
- Features: 23
- Target variable: `price`

The dataset contains numerical, categorical, and multi-valued vehicle specification variables.

### Important Features

**Numerical features:**
- `km`
- `Gears`
- `age`
- `Previous_Owners`
- `hp_kW`
- `Inspection_new`
- `Displacement_cc`
- `Weight_kg`
- `cons_comb`

**Categorical features:**
- `make_model`
- `body_type`
- `vat`
- `Type`
- `Fuel`
- `Paint_Type`
- `Upholstery_type`
- `Gearing_Type`
- `Drive_chain`

**Target variable:** `price`

## Project Workflow

### 1. Data Understanding and Cleaning
- Loaded and examined the dataset.
- Analysed data types and feature distributions.
- Checked missing values and duplicate records.
- Removed 1,673 duplicate observations.
- Analysed categorical and numerical variables.
- Investigated rare categories.

### 2. Exploratory Data Analysis (EDA)
- Analysed numerical feature distributions.
- Examined categorical frequency distributions.
- Studied the distribution of vehicle prices.
- Performed correlation analysis.
- Compared average prices across vehicle categories.
- Identified potential outliers using the IQR method.

### 3. Feature Engineering
- Created `age_group` to represent vehicle age categories.
- Created `km_per_year` to measure average annual mileage.
- Created `hp_per_litre` to represent engine power relative to engine displacement.
- Analysed multi-valued vehicle specification columns.
- Prepared meaningful numerical representations of the available predictors.

### 4. Target Transformation
The original vehicle price distribution was positively skewed. Therefore, a logarithmic transformation was applied:

`log_price = log1p(price)`

This reduced skewness and the influence of extreme prices.

### 5. Feature Encoding and Scaling
- Applied one-hot encoding to categorical predictors.
- Used `drop_first=True` to avoid the dummy variable trap.
- Split the dataset into training and testing sets using a 70:30 ratio.
- Used `random_state=42` for reproducibility.
- Applied StandardScaler to numerical predictor columns.
- Fitted the scaler on training data and transformed both training and testing data.

### 6. Model Development

Three regression models were developed:

**Linear Regression**
- Established the baseline predictive model.
- Evaluated training and testing performance.

**Ridge Regression**
- Applied L2 regularisation.
- Used cross-validation for alpha selection.
- Optimal alpha: 9.

**Lasso Regression**
- Applied L1 regularisation.
- Used cross-validation for alpha selection.
- Optimal alpha: 0.0002.
- Analysed coefficient shrinkage and feature selection.

### 7. Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Additional analysis included:
- Actual vs Predicted Price plot.
- Error vs Alpha plots.
- Coefficient comparison across models.
- Top 10 influential Lasso coefficients.
- Features eliminated by Lasso.

## Model Performance

The following results were obtained on the held-out testing dataset. MAE, MSE, and RMSE are calculated on the log-transformed target scale.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.085877 | 0.116356 | 0.914667 |
| Ridge Regression | 0.085859 | 0.116348 | 0.914679 |
| Lasso Regression | 0.085841 | 0.116337 | 0.914694 |

### Optimal Hyperparameters

- Ridge Regression: Alpha = 9
- Lasso Regression: Alpha = 0.0002

### Baseline Model – Original Price Scale

After converting predictions back to the original price scale using `expm1()`:

- MAE: €1,608.30
- RMSE: €2,496.26
- R²: 0.8850

## Key Findings

1. Vehicle engine power (`hp_kW`) has a strong positive relationship with predicted vehicle price.
2. Vehicle age (`age`) and mileage (`km`) have negative relationships with predicted price.
3. Vehicle model is an important predictor of used-car prices.
4. Ridge Regression provides coefficient shrinkage and helps manage multicollinearity.
5. Lasso Regression provides automatic feature selection through L1 regularisation.
6. Lasso achieved the lowest test MAE and RMSE and the highest R² among the three evaluated models.
7. The predictive performance differences between the three models are marginal.

## Conclusion

The project demonstrates the application of advanced regression techniques for used-car price prediction.

All three models achieved strong and similar predictive performance. Lasso Regression performed marginally better based on the test evaluation metrics and provided the additional benefit of feature selection.

The final model can serve as a data-driven pricing reference for estimating used-car values. Actual market prices may also depend on factors such as vehicle condition, demand, negotiation, and other variables not captured in the dataset.

## Technologies and Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Google Colab
- Git and GitHub

## Repository Structure

```text
Used-Car-Price-Prediction-Advanced-Regression/
│
├── data/
│   └── Car_Price_data.csv
│
├── notebooks/
│   └── Advanced_Regression_Car_Price_Prediction.ipynb
│
├── images/
│   ├── price_distribution.png
│   ├── actual_vs_predicted.png
│   ├── ridge_alpha_tuning.png
│   ├── lasso_alpha_tuning.png
│   └── lasso_coefficients.png
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Used-Car-Price-Prediction-Advanced-Regression.git
```

### 2. Navigate to the project directory

```bash
cd Used-Car-Price-Prediction-Advanced-Regression
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook in the `notebooks` folder and execute the cells in sequence.

## Author

**Swati J. Nimbolkar**

AI & Machine Learning | Data Science | Regression Modelling

## Acknowledgements

This project was completed as part of an Advanced Regression assignment, applying statistical learning and regularisation techniques to a real-world used-car price prediction problem.

