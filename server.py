import os
from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def identify_distribution(series):
    skew = series.skew()
    kurt = series.kurtosis()
    if abs(skew) < 0.5:
        return 'normal'
    elif skew > 1:
        return 'right_skewed'
    elif skew < -1:
        return 'left_skewed'
    else:
        return 'moderately_skewed'

def handle_missing_values(df, summary):
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        pct = round(missing_count / len(df) * 100, 2)
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            summary['imputations'][col] = {'method': f"mode ({mode_val})", 'missing_pct': pct}
        else:
            dist = identify_distribution(df[col].dropna())
            if dist == 'normal':
                val = df[col].mean()
                df[col].fillna(val, inplace=True)
                summary['imputations'][col] = {'method': f"mean ({val:.2f})", 'missing_pct': pct}
            else:
                val = df[col].median()
                df[col].fillna(val, inplace=True)
                summary['imputations'][col] = {'method': f"median ({val:.2f})", 'missing_pct': pct}
    return df

def treat_outliers(df, summary, method='cap'):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = len(df[(df[col] < lower) | (df[col] > upper)])
        if outlier_count > 0:
            if method == 'cap':
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])
                summary['outliers'][col] = {'action': f"capped to [{lower:.2f}, {upper:.2f}]", 'count': outlier_count}
    return df

def perform_eda(df):
    summary = {
        'shape': list(df.shape),
        'numerical': [],
        'categorical': [],
        'distributions': {},
        'imputations': {},
        'outliers': {},
        'missing': {}
    }
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing > 0:
            summary['missing'][col] = {'count': missing, 'pct': round(missing/len(df)*100,2)}
        if df[col].dtype == 'object':
            summary['categorical'].append(col)
        else:
            summary['numerical'].append(col)
            summary['distributions'][col] = identify_distribution(df[col].dropna())
    return summary

def get_numerical_stats(df, numerical_cols):
    stats_data = {}
    for col in numerical_cols:
        s = df[col].dropna()
        stats_data[col] = {
            'mean': round(float(s.mean()), 4),
            'median': round(float(s.median()), 4),
            'std': round(float(s.std()), 4),
            'min': round(float(s.min()), 4),
            'max': round(float(s.max()), 4),
            'skew': round(float(s.skew()), 4),
            'kurtosis': round(float(s.kurtosis()), 4),
            'q25': round(float(s.quantile(0.25)), 4),
            'q75': round(float(s.quantile(0.75)), 4),
            'histogram': {
                'values': s.tolist()[:500],  # sample for viz
            }
        }
    return stats_data

def get_categorical_stats(df, categorical_cols):
    cat_data = {}
    for col in categorical_cols:
        vc = df[col].value_counts()
        cat_data[col] = {
            'unique': int(df[col].nunique()),
            'top_values': {str(k): int(v) for k, v in vc.head(15).items()},
            'mode': str(df[col].mode()[0]) if not df[col].mode().empty else 'N/A'
        }
    return cat_data

def get_correlation_matrix(df, numerical_cols):
    if len(numerical_cols) < 2:
        return {}
    corr = df[numerical_cols].corr().round(3)
    return {
        'columns': list(corr.columns),
        'matrix': corr.values.tolist()
    }

def perform_feature_engineering(df, summary):
    original_cols = list(df.columns)
    fe_log = []
    numerical_cols = summary['numerical']
    categorical_cols = summary['categorical']

    # 1. Interaction features (top numerical pairs)
    if len(numerical_cols) >= 2:
        pairs = [(numerical_cols[i], numerical_cols[j])
                 for i in range(min(3, len(numerical_cols)))
                 for j in range(i+1, min(4, len(numerical_cols)))]
        for col1, col2 in pairs:
            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1] * df[col2]
            fe_log.append({'feature': new_col, 'type': 'Interaction', 'formula': f"{col1} × {col2}"})

    # 2. Ratio features
    if len(numerical_cols) >= 2:
        col1, col2 = numerical_cols[0], numerical_cols[1]
        new_col = f"{col1}_div_{col2}"
        df[new_col] = df[col1] / (df[col2] + 1e-9)
        fe_log.append({'feature': new_col, 'type': 'Ratio', 'formula': f"{col1} / {col2}"})

    # 3. Polynomial features for top numerical cols
    for col in numerical_cols[:3]:
        sq_col = f"{col}_squared"
        df[sq_col] = df[col] ** 2
        fe_log.append({'feature': sq_col, 'type': 'Polynomial', 'formula': f"{col}²"})

        cube_col = f"{col}_cubed"
        df[cube_col] = df[col] ** 3
        fe_log.append({'feature': cube_col, 'type': 'Polynomial', 'formula': f"{col}³"})

    # 4. Log transform for skewed columns
    for col in numerical_cols:
        dist = summary['distributions'].get(col, '')
        if 'skewed' in dist and df[col].min() > 0:
            log_col = f"{col}_log"
            df[log_col] = np.log1p(df[col])
            fe_log.append({'feature': log_col, 'type': 'Log Transform', 'formula': f"log(1 + {col})"})

    # 5. Aggregation features for categorical + numerical
    for cat_col in categorical_cols[:2]:
        for num_col in numerical_cols[:2]:
            group = df.groupby(cat_col)[num_col]
            mean_col = f"{cat_col}_{num_col}_mean"
            std_col = f"{cat_col}_{num_col}_std"
            df[mean_col] = df[cat_col].map(group.mean())
            df[std_col] = df[cat_col].map(group.std().fillna(0))
            fe_log.append({'feature': mean_col, 'type': 'Group Aggregation', 'formula': f"mean({num_col}) by {cat_col}"})
            fe_log.append({'feature': std_col, 'type': 'Group Aggregation', 'formula': f"std({num_col}) by {cat_col}"})

    # 6. Binning for top numerical columns
    for col in numerical_cols[:2]:
        bin_col = f"{col}_bin"
        try:
            df[bin_col] = pd.qcut(df[col], q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop').astype(str)
            fe_log.append({'feature': bin_col, 'type': 'Binning', 'formula': f"quantile bins of {col}"})
        except:
            pass

    # 7. Label encode new categorical + original categorical
    all_cat = [c for c in df.columns if df[c].dtype == 'object']
    for col in all_cat:
        enc_col = f"{col}_encoded"
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[col].astype(str))
        fe_log.append({'feature': enc_col, 'type': 'Label Encoding', 'formula': f"LabelEncoder({col})"})

    new_features = [f['feature'] for f in fe_log]
    return df, fe_log, new_features

def suggest_scaling_methods(df, summary):
    suggestions = {}
    for col in summary['numerical']:
        dist = summary['distributions'].get(col, 'skewed')
        if dist == 'normal':
            suggestions[col] = {'method': 'StandardScaler (Z-score)', 'reason': 'Normally distributed'}
        elif 'skewed' in dist:
            suggestions[col] = {'method': 'RobustScaler / Log Transform', 'reason': 'Skewed distribution'}
        else:
            suggestions[col] = {'method': 'MinMaxScaler', 'reason': 'Bounded range preferred'}
    return suggestions

def suggest_encoding(df, summary):
    suggestions = {}
    for col in summary['categorical']:
        cardinality = df[col].nunique()
        if cardinality == 2:
            suggestions[col] = {'method': 'Binary Encoding', 'cardinality': cardinality}
        elif cardinality <= 10:
            suggestions[col] = {'method': 'One-Hot Encoding', 'cardinality': cardinality}
        elif cardinality <= 50:
            suggestions[col] = {'method': 'Ordinal / Target Encoding', 'cardinality': cardinality}
        else:
            suggestions[col] = {'method': 'Target Encoding / Embedding', 'cardinality': cardinality}
    return suggestions

def recommend_algorithms(df):
    recommendation = {}
    target_col = df.columns[-1]
    target = df[target_col]
    if target.nunique() <= 1:
        return {'error': 'Target column is constant or invalid.'}
    target_type = type_of_target(target)
    if target_type in ['binary', 'multiclass']:
        recommendation['problem_type'] = 'classification'
        recommendation['algorithms'] = [
            {'name': 'Logistic Regression', 'use_case': 'Baseline, interpretable'},
            {'name': 'Random Forest Classifier', 'use_case': 'High accuracy, handles non-linearity'},
            {'name': 'SVM', 'use_case': 'Effective in high-dimensional spaces'},
            {'name': 'Gradient Boosting', 'use_case': 'Strong ensemble method'},
            {'name': 'XGBoost', 'use_case': 'State-of-art for tabular data'}
        ]
    elif target_type == 'continuous':
        recommendation['problem_type'] = 'regression'
        recommendation['algorithms'] = [
            {'name': 'Linear Regression', 'use_case': 'Baseline, interpretable'},
            {'name': 'Random Forest Regressor', 'use_case': 'Robust, handles outliers'},
            {'name': 'XGBoost Regressor', 'use_case': 'State-of-art for tabular data'},
            {'name': 'SVR', 'use_case': 'Effective with kernel tricks'}
        ]
    else:
        recommendation['problem_type'] = 'unknown'
        recommendation['algorithms'] = []
    recommendation['target_column'] = target_col
    recommendation['target_type'] = target_type
    if target_type in ['binary', 'multiclass']:
        recommendation['class_balance'] = {str(k): int(v) for k, v in target.value_counts().items()}
    else:
        recommendation['class_balance'] = None
    return recommendation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV: {str(e)}'}), 500

    summary = perform_eda(df)
    numerical_stats = get_numerical_stats(df, summary['numerical'])
    categorical_stats = get_categorical_stats(df, summary['categorical'])
    correlation = get_correlation_matrix(df, summary['numerical'])

    df = handle_missing_values(df, summary)
    df = treat_outliers(df, summary)

    df_engineered, fe_log, new_features = perform_feature_engineering(df, summary)

    scaling = suggest_scaling_methods(df, summary)
    encoding = suggest_encoding(df, summary)
    ml_recommendation = recommend_algorithms(df_engineered)

    cleaned_path = os.path.join(STATIC_FOLDER, 'cleaned_dataset.csv')
    df_engineered.to_csv(cleaned_path, index=False)

    return jsonify({
        'summary': summary,
        'numerical_stats': numerical_stats,
        'categorical_stats': categorical_stats,
        'correlation': correlation,
        'scaling_suggestions': scaling,
        'encoding_suggestions': encoding,
        'ml_recommendation': ml_recommendation,
        'feature_engineering': fe_log,
        'new_features_count': len(new_features),
        'original_features_count': len(summary['numerical']) + len(summary['categorical']),
        'final_shape': list(df_engineered.shape),
        'download_link': '/download'
    })

@app.route('/download', methods=['GET'])
def download_file():
    return send_file(os.path.join(STATIC_FOLDER, 'cleaned_dataset.csv'), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)