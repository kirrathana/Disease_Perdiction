# Heart Disease Prediction System

A production-quality machine learning project that predicts heart disease risk based on patient medical information. Built with Python, Scikit-learn, XGBoost, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit-learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)

---

## Objective

Build an end-to-end machine learning pipeline that:

- Analyzes heart disease medical data through exploratory data analysis
- Trains and compares multiple classification models
- Selects the best model based on ROC-AUC performance
- Deploys an interactive Streamlit web application for real-time predictions

The target variable is **heart disease presence**:
- `0` = No Heart Disease
- `1` = Heart Disease

---

## Features

- **Automated Data Analysis** — Dataset inspection, missing value handling, and duplicate removal
- **Exploratory Data Analysis** — Jupyter notebook with visualizations (histograms, boxplots, correlation heatmaps, pairplots)
- **Multiple ML Models** — Logistic Regression, Random Forest, SVM, and XGBoost
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1 Score, ROC-AUC, confusion matrix, ROC curve
- **Feature Importance** — Random Forest feature importance analysis
- **Interactive Web App** — Streamlit UI with prediction gauge, probability charts, and patient input forms
- **Model Persistence** — Saved model, scaler, and metadata using joblib

---

## Dataset Information

**Source:** UCI Machine Learning Repository — Heart Disease Dataset

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Sex (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (0/1) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (0/1) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia (1 = normal, 2 = fixed, 3 = reversible) |
| `target` | Heart disease (0 = No, 1 = Yes) |

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **Scikit-Learn** | Machine learning models and preprocessing |
| **XGBoost** | Gradient boosting classifier |
| **Streamlit** | Interactive web application |
| **Plotly** | Interactive charts and gauges |
| **Joblib** | Model serialization and persistence |

---

## Machine Learning Models

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear model for binary classification |
| **Random Forest** | Ensemble of decision trees |
| **SVM** | Support Vector Machine with RBF kernel |
| **XGBoost** | Gradient boosting framework |

The best model is selected based on **ROC-AUC** score.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve |

---

## Folder Structure

```
Disease_Prediction/
│
├── dataset/
│   └── heart.csv              # Heart disease dataset
│
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis notebook
│
├── models/                    # Generated after training
│   ├── best_model.pkl         # Best trained model
│   ├── scaler.pkl             # Feature scaler
│   ├── model_metadata.pkl     # Model metrics and metadata
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_histograms.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
│
├── app.py                     # Streamlit web application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Installation

1. **Clone or download the project**

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Train Model

Run the training script to analyze data, train models, evaluate performance, and save artifacts:

```bash
python train_model.py
```

This will:
- Analyze the dataset and print summary statistics
- Generate EDA plots in `models/`
- Train Logistic Regression, Random Forest, SVM, and XGBoost
- Print a formatted comparison table
- Save the best model, scaler, and evaluation plots

---

## Run Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the prediction interface.

---

## Screenshots Section

After training and running the app, you will see:

- **Prediction Page** — Patient input form with risk assessment and probability gauge
- **Model Info Page** — Model metrics, comparison table, and evaluation plots
- **EDA Plots** — Target distribution, correlation heatmap, feature histograms in `models/`

---

## Future Enhancements

- Diabetes prediction module
- Multiple disease prediction system
- Deep learning integration (neural networks)
- Explainable AI (SHAP, LIME)
- Cloud deployment (AWS, GCP, Azure)
- REST API endpoint
- User authentication and prediction history
- Mobile-responsive design improvements

---

## License

This project is for educational purposes. Please consult healthcare professionals for medical decisions.

---

## Author

Built as an internship-level, resume-worthy machine learning project demonstrating full ML pipeline development.
