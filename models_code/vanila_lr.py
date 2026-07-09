"""
Vanilla Logistic Regression model for CFPB data.

Model is performed to compare the result with Balanced Random Forest Classifier, 
XGBoost, CatBoost and Dummy Classifier. 
"""

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

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
)-> LogisticRegression:
    """
    Training and testing Logistic Regression.

    Parameters:
    X_train : feature DataFrame for training
    X_test : feature DataFrame for test
    y_train : target Series for training
    y_test : target Series for testing
    """

    with mlflow.start_run(run_name="logistic_regression"):
        categories = X_train.columns.to_list()
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categories)
            ],
            remainder="passthrough"
        )
        
        lr_model = Pipeline(
            steps=[
                    ("preprocess", preprocessor),
                    ("classifier", LogisticRegression(
                        class_weight = "balanced",
                        solver= "lbfgs",
                        random_state= 42,
                        max_iter= 1000
                    ))
            ]
        )

        lr_model.fit(X_train, y_train)

        y_pred_lr = lr_model.predict(X_test)
        
        f1_macro_lr = f1_score(y_test, y_pred_lr, average="macro")
        mlflow.log_metric("f1_macro", f1_macro_lr)
        
        mlflow.log_text(
            classification_report(y_test, y_pred_lr),
            "classification_report.txt"
        )

        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("Training Info", "Basic Logistic Regression model for CFPB data")

        return lr_model
    
def main() -> None:
    """Run Basic Logistic Regression classification for CFPB data."""
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
