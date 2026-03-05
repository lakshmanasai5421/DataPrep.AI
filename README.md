## AutoEDA Automatic EDA and preprocessing tool using Machine Learning

This project is a smart data assistant that helps you clean your CSV datasets, perform basic EDA, and even get machine learning advice — all from a simple web interface built with Flask.

---

## What It Does

✅ Upload your own CSV file
✅ Automatically detect numerical and categorical columns
✅ Clean missing values intelligently:

* Mean for normal distributions
* Median for skewed data
* Mode for categorical columns

✅ Handle outliers using the IQR method:

* Capping extreme values within acceptable range (Winsorization)

✅ EDA summary:

* Identify data types
* Understand distributions

✅ Suggest the right preprocessing tools:

* `StandardScaler` for normal distributions
* `MinMaxScaler`, `RobustScaler`, or log transformation for skewed features

✅ Recommend encoding strategies:

* One-Hot Encoding for simple categories
* Ordinal or Target Encoding for complex categories

✅ Recommend ML algorithms:

* Classify the task as regression or classification
* Suggest algorithms based on data size and target characteristics

✅ Download a cleaned version of your dataset
✅ Interact through a clean, modern frontend UI

---

## Technologies Used

* **Backend**: Python, Flask, Pandas, NumPy, Scikit-learn
* **Frontend**: HTML + Bootstrap
* Easily deployable on Heroku, Render, or Docker

---

## Project Structure

```
eda-cleaner-app/
├── app.py                  # Flask backend logic
├── uploads/                # Where uploaded CSVs are saved
├── static/                 # Where cleaned CSVs are stored
├── templates/
│   └── index.html          # Web interface
└── README.md               # This file
```

---


