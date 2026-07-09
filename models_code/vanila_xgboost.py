"""
Vanilla XGBoost model for CFPB data.

Model is performed to compare the result with Balanced Random Forest Classifier, 
Logistic Regression, CatBoost and Dummy Classifier. 
"""

import mlflow
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder



DATA_URL = "https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/processed/processed_data.parquet"
TARGET_COLUMN = "Company response to consumer"
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "cfpb_models"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(url: str, target: str):
    """
    Load parquet file and return features and target.

    Parameters:
    url    : remote path to parquet file
    target : name of the column to predict

    Returns:
    features : pd.DataFrame
    labels   : pd.Series    
    """
    df = pd.read_parquet(url)
    features = df.drop(columns= [target]) 
    labels = df[target]
    return features, labels

def split_data(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size:float = TEST_SIZE,
    random_state:int = RANDOM_STATE
):
    """
    Stratified spliting data into: train, test.

    Parameters:
    features     : feature DataFrame
    labels       : target Series
    test_size    : size of the test
    random_state : reproducibility seed

    Returns:
    X_train, X_test,
    y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size= test_size,
        random_state= random_state,
        stratify= labels
    )

    return (
        X_train, X_test,
        y_train, y_test
    )

def model_run(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
)-> XGBClassifier:
    """
    Training and testing Logistic Regression.

    Parameters:
    X_train : feature DataFrame for training
    X_test : feature DataFrame for test
    y_train : target Series for training
    y_test : target Series for testing
    """
    with mlflow.start_run(run_name= "vanilla_xgboost"):

        xgb_model = XGBClassifier(
            tree_method = "hist",
            enable_categorical=True, 
            device="cuda",
            eval_metric="mlogloss",
            random_state=42,
            verbosity = 2)
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_train)
        
        xgb_model.fit(X_train, y_encoded)

        y_pred = label_encoder.inverse_transform(xgb_model.predict(X_test))

        f1_macro = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("f1_macro", f1_macro)

        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")

        mlflow.set_tag("Training Info", "Vanilla XGBoost model for CFPB data")
        mlflow.set_tag("model_type", "XGBoost")
    return xgb_model

def main() -> None:
    """Run XGBoost classification for CFPB data."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    features, labels = load_data(DATA_URL, TARGET_COLUMN)

    (
    X_train, X_test,
    y_train, y_test
    ) = split_data(features, labels)

    model_run(
    X_train,
    X_test,
    y_train,
    y_test,
    )

if __name__ == "__main__":
    main()


