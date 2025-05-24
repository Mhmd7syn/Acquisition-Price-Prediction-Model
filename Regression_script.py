from Modules import Preprocessing, AddressParser, DateParser, Encoding, Merge, Text
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt


def preprocessing(df_merged, df_members, folder_path):
    Preprocessing.renaming(df_merged, df_members)
    AddressParser.fill_missing_locations(df_merged)
    
    DateParser.clean_date_columns(df_merged)
    DateParser.date_validate(df_merged)

    Encoding.numerical_reformatting(df_merged)
    Encoding.ordinal_features(df_merged)
    df_merged = Encoding.multipleFeatures(df_merged)

    Preprocessing.null_numerical(df_merged, folder_path)
    df_merged = Text.tf_idf(df_merged, folder_path)

    df_merged = Preprocessing.feature_engineering(df_merged)
    
    return df_merged


def normalization(df):
    scaler = load(folder_path + 'Models\\' + "minmax_scaler.joblib")
    features_to_scale = load(folder_path + 'Models\\' + "features_to_scale.joblib")

    # Scale all features except 'Price'
    scaled_features = scaler.transform(df[features_to_scale])

    # Create a new DataFrame with scaled features
    df_scaled = pd.DataFrame(scaled_features, 
                            columns=features_to_scale, 
                            index=df.index)

    # Add back the unscaled 'Price' column
    df_scaled['Price'] = df['Price']

    return df_scaled


def plot_best_scores(df_scaled):
    models = {
        "Linear Regression": {
            "features_file": 'best_features_Linear.csv',
            "model_file": 'linear_regression_model.joblib',
            "color": '#1f77b4'
        },
        "Polynomial Regression": {
            "features_file": 'best_features_Poly.csv',
            "model_file": 'Poly_regression_model.joblib',
            "color": '#ff7f0e'
        },
        "Random Forest": {
            "features_file": 'best_features_rf.csv',
            "model_file": 'Random_Forest_model.joblib',
            "color": '#2ca02c'
        },
        "XGBoost": {
            "features_file": 'best_features_XGBOOST.csv',
            "model_file": 'xgb_model.joblib',
            "color": '#d62728'
        }
    }

    y = df_scaled['Price']
    scores = []
    model_names = []
    
    for model_name, config in models.items():
        # Load selected features
        best_features = pd.read_csv(folder_path + 'CSV Files\\' + config['features_file'])
        selected_features = [f.strip().strip("'") for f in 
                           best_features.iloc[0]['features'].strip("()").split(",")]
        
        # Prepare data and load model
        X = df_scaled[selected_features]
        model = load(folder_path + 'Models\\' + config['model_file'])
        
        # Store score for plotting
        test_score = model.score(X, y)
        scores.append(test_score)
        model_names.append(model_name)

    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, color=[config['color'] for config in models.values()])
    
    # Add score labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    # Format plot
    plt.title('Model Performance Comparison (R² Scores)', pad=20)
    plt.ylabel('R² Score')
    plt.ylim(0, 1.05)  # R² typically ranges 0-1
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_script():
    df_merged, df_members = Merge.merge(folder_path)
    df_merged = preprocessing(df_merged, df_members, folder_path)
    df_scaled = normalization(df_merged)
    plot_best_scores(df_scaled)

folder_path = "C:\\Users\\HP\\OneDrive - Faculty of Computer and Information Sciences (Ain Shams University)\\ML_Project\\Test_Script\\"
run_script()