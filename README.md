# UrbanFare : Dynamic Taxi Trip Prediction

UrbanFare is an end-to-end machine learning system designed to provide dynamic, high-precision taxi fare predictions. By capturing complex interactions across distance, trip duration, time of day, day of week, traffic density, weather, and pricing rates, the platform delivers accurate, real-time trip price estimations. The system incorporates data cleaning, exploratory data analysis (EDA), feature engineering, hyperparameter optimization across 8 machine learning models, and interactive web deployment via Streamlit.

 **Live Streamlit App:** [UrbanFare Web Application](https://taxi--trip---prediction-w6hdlacgeqzukffzqz8ixg.streamlit.app/)[cite: 1]

---

##  Problem Overview

Taxi fares vary dynamically based on physical metrics, base charges, and environmental variables like traffic congestion and inclement weather[cite: 1]. Traditional linear estimations often fail to model non-linear price relationships across these factors accurately[cite: 1].

UrbanFare addresses this challenge by utilizing an optimized ensemble regression model that captures feature dependencies, yielding highly reliable price forecasts ($MAE = 4.95$, $Test\ R^2 = 0.9503$)[cite: 1].

---

##  Dataset Overview

* **Total Records:** 1,000[cite: 1]
* **Attributes:** 10 Input Features | 1 Target Variable (`Trip_Price`)[cite: 1]
* **Data Quality:** ~5% missing values per feature (~49 rows dropped due to missing target)[cite: 1]; 0 duplicates[cite: 1].

### Feature Schema

| Feature Name | Type | Description / Levels |
| :--- | :--- | :--- |
| `Trip_Distance_km` | Numerical | Trip distance in kilometers (right-skewed: 2.26)[cite: 1] |
| `Trip_Duration_Minutes` | Numerical | Trip duration in minutes[cite: 1] |
| `Base_Fare` | Numerical | Baseline fare rate[cite: 1] |
| `Per_Km_Rate` | Numerical | Per-kilometer charge rate[cite: 1] |
| `Per_Minute_Rate` | Numerical | Per-minute charge rate[cite: 1] |
| `Passenger_Count` | Numerical | Number of passengers in vehicle[cite: 1] |
| `Day_of_Week` | Binary | `Weekday` / `Weekend`[cite: 1] |
| `Time_of_Day` | Categorical | Morning, Afternoon, Evening, Night (4 levels)[cite: 1] |
| `Traffic_Conditions` | Categorical | Low, Medium, High (3 levels)[cite: 1] |
| `Weather` | Categorical | Clear, Rainy, Snowy (3 levels)[cite: 1] |

---

##  Data Preprocessing & Pipeline Architecture

1. **Target Analysis:** `Trip_Price` displayed right-skewness (skew = 3.73, mean = 56.87, median = 50.07)[cite: 1].
2. **Missing Value Imputation:** Numerical features were imputed with medians and categorical features with modes using `SimpleImputer` fit exclusively on training data to prevent leakage[cite: 1].
3. **Outlier Treatment:** Applied IQR threshold capping on extreme distance values ($> 100\text{ km}$ capped at upper bound $74.50\text{ km}$) rather than discarding records[cite: 1].
4. **Encoding & Scaling:** One-Hot Encoding (`drop='first'`) expanded categorical features into 14 input attributes[cite: 1]. Linear models utilized `StandardScaler` and `log1p` target transformations, while tree models trained on unscaled data[cite: 1].

---

##  Model Evaluation & Benchmarks

Models were trained and evaluated on an 80/20 split (760 train / 191 test) using 15-fold cross-validation[cite: 1]. Ensemble architectures significantly outperformed traditional linear baselines[cite: 1].

| Model Algorithm | CV $R^2$ | Test $R^2$ | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Tuned)**  | **0.9425** | **0.9503** | **4.95** | **10.78** |
| **XGBoost (Tuned)** | 0.9730 | 0.9413 | 5.14 | 11.46 |
| **Random Forest** | 0.9270 | 0.9370 | 6.29 | 12.14 |
| **Decision Tree** | 0.8680 | 0.8800 | 9.97 | 16.74 |
| **Ridge Regression** | 0.8690 | 0.8850* | 12.14* | 17.18* |
| **Lasso Regression** | 0.8710 | 0.8840* | 12.13* | 17.18* |
| **Linear Regression** | 0.8690 | 0.8080* | 9.76* | 21.21* |
| **AdaBoost** | 0.8420 | 0.8740 | 12.12 | 17.17 |

*\*Linear model evaluation metrics reflect target log-scale transformation[cite: 1].*

### Optimal Hyperparameters (Gradient Boosting)
* `n_estimators`: 200[cite: 1]
* `learning_rate`: 0.05[cite: 1]
* `max_depth`: 5[cite: 1]
* `min_samples_leaf`: 3[cite: 1]
* `min_samples_split`: 10[cite: 1]
* `subsample`: 0.8[cite: 1]

---

## 🚀 Web Deployment

The final tuned Gradient Boosting model was serialized via Joblib (`gradient_boosting_tuned.pkl`) and wrapped inside a Streamlit web application[cite: 1]. The app accepts user parameters, automatically applies preprocessing pipelines, and calculates instant fare predictions[cite: 1].

---

##  Repository Structure

```text
├── data/
│   └── taxi_trip_data.csv          # Raw trip dataset
├── models/
│   └── gradient_boosting_tuned.pkl  # Trained model artifact
├── app.py                          # Streamlit web entry point
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Environment exclusion rules
└── README.md                       # Documentation
