# Machine-Learning-Predictive-Rainfall-Analytics

## Project Overview

This project is a machine learning pipeline designed to predict the "RainTomorrow" indicator using a weather dataset. The dataset contains various meteorological features collected from different locations in Australia. The pipeline includes data preprocessing, feature selection, model training, and evaluation using several machine learning algorithms.

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Selection](#feature-selection)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Results](#results)

## Installation

Run it on Jupyter Notebook

## Dataset

The dataset used in this project is `weatherAUS.csv`. It contains various features like temperature, humidity, wind speed, and others. The target variable is `RainTomorrow`, which indicates whether it will rain the next day.

## Data Preprocessing

1. **Data Loading:**
   ```python
   full_data = pd.read_csv("weatherAUS.csv")
   full_data['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
   full_data['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
   ```

2. **Handling Class Imbalance:**
   ```python
   no = full_data[full_data.RainTomorrow == 0]
   yes = full_data[full_data.RainTomorrow == 1]
   yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
   oversampled = pd.concat([no, yes_oversampled])
   ```

3. **Handling Missing Values:**
   ```python
   oversampled['Date'] = oversampled['Date'].fillna(oversampled['Date'].mode()[0])
   oversampled['Location'] = oversampled['Location'].fillna(oversampled['Location'].mode()[0])
   oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(oversampled['WindGustDir'].mode()[0])
   oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(oversampled['WindDir9am'].mode()[0])
   oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(oversampled['WindDir3pm'].mode()[0])
   ```

4. **Label Encoding:**
   ```python
   lencoders = {}
   for col in oversampled.select_dtypes(include=['object']).columns:
       lencoders[col] = LabelEncoder()
       oversampled[col] = lencoders[col].fit_transform(oversampled[col])
   ```

5. **Iterative Imputation:**
   ```python
   mice_imputer = IterativeImputer(estimator=LinearRegression(), max_iter=5, random_state=0)
   MiceImputed = oversampled.copy(deep=True)
   MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)
   ```

6. **Outlier Detection and Removal:**
   ```python
   Q1 = MiceImputed.quantile(0.25)
   Q3 = MiceImputed.quantile(0.75)
   IQR = Q3 - Q1
   MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) | (MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]
   ```

## Feature Selection

1. **Filter Method (Chi-Square):**
   ```python
   X = modified_data.loc[:, modified_data.columns != 'RainTomorrow']
   y = modified_data[['RainTomorrow']]
   selector = SelectKBest(chi2, k=10)
   selector.fit(X, y)
   X_new = selector.transform(X)
   print(X.columns[selector.get_support(indices=True)])
   ```

2. **Model-Based Selection (Random Forest):**
   ```python
   selector = SelectFromModel(rf(n_estimators=100, random_state=0))
   selector.fit(X, y)
   support = selector.get_support()
   features = X.loc[:, support].columns.tolist()
   ```

## Model Training and Evaluation

The following models were trained and evaluated:

1. **Logistic Regression**
   ```python
   params_lr = {'penalty': 'l1', 'solver': 'liblinear'}
   model_lr = LogisticRegression(**params_lr)
   model_lr, accuracy_lr, roc_auc_lr, coh_kap_lr, tt_lr = run_model(model_lr, X_train, y_train, X_test, y_test)
   ```

2. **Decision Tree**
   ```python
   params_dt = {'max_depth': 16, 'max_features': "sqrt"}
   model_dt = DecisionTreeClassifier(**params_dt)
   model_dt, accuracy_dt, roc_auc_dt, coh_kap_dt, tt_dt = run_model(model_dt, X_train, y_train, X_test, y_test)
   ```

3. **Neural Network**
   ```python
   params_nn = {'hidden_layer_sizes': (30, 30, 30), 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 500}
   model_nn = MLPClassifier(**params_nn)
   model_nn, accuracy_nn, roc_auc_nn, coh_kap_nn, tt_nn = run_model(model_nn, X_train, y_train, X_test, y_test)
   ```

4. **Random Forest**
   ```python
   params_rf = {'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 12345}
   model_rf = RandomForestClassifier(**params_rf)
   model_rf, accuracy_rf, roc_auc_rf, coh_kap_rf, tt_rf = run_model(model_rf, X_train, y_train, X_test, y_test)
   ```

5. **LightGBM**
   ```python
   params_lgb = {'colsample_bytree': 0.95, 'max_depth': 16, 'min_split_gain': 0.1, 'n_estimators': 200, 'num_leaves': 50, 'reg_alpha': 1.2, 'reg_lambda': 1.2, 'subsample': 0.95, 'subsample_freq': 20}
   model_lgb = lgb.LGBMClassifier(**params_lgb)
   model_lgb, accuracy_lgb, roc_auc_lgb, coh_kap_lgb, tt_lgb = run_model(model_lgb, X_train, y_train, X_test, y_test)
   ```

6. **CatBoost**
   ```python
   params_cb = {'iterations': 50, 'max_depth': 16}
   model_cb = cb.CatBoostClassifier(**params_cb)
   model_cb, accuracy_cb, roc_auc_cb, coh_kap_cb, tt_cb = run_model(model_cb, X_train, y_train, X_test, y_test, verbose=False)
   ```

7. **XGBoost**
   ```python
   params_xgb = {'n_estimators': 500, 'max_depth': 16}
   model_xgb = xgb.XGBClassifier(**params_xgb)
   model_xgb, accuracy_xgb, roc_auc_xgb, coh_kap_xgb, tt_xgb = run_model(model_xgb, X_train, y_train, X_test, y_test)
   ```

## Results

The models were evaluated using various metrics including accuracy, ROC AUC, Cohen's Kappa, and time taken for training. Here are the key results:

- **Logistic Regression:** Accuracy = 0.789, ROC AUC = 0.863
- **Decision Tree:** Accuracy = 0.863, ROC AUC = 0.901
- **Neural Network:** Accuracy = 0.887, ROC AUC = 0.958
- **Random Forest:** Accuracy = 0.930, ROC AUC = 0.979
- **LightGBM:** Accuracy = 0.929, ROC AUC = 0.978
- **CatBoost:** Accuracy = 0.913, ROC AUC = 0.964
- **XGBoost:** Accuracy = 0.923, ROC AUC = 0.973
