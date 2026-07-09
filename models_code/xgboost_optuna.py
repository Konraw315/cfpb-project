"""
CFPB complaint classification pipeline.
 
XGBoost classifier with Optuna hyperparameter tuning,
MLflow experiment tracking.
"""

import os
import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
 

DATA_URL = "https://cfpb-project-742435394310-us-east-1-an.s3.us-east-1.amazonaws.com/processed/processed_data.parquet"
TARGET_COLUMN = "Company response to consumer"
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "cfpb_models"
OPTUNA_STORAGE = "sqlite:///optuna_study.db"
STUDY_NAME = "cfpb_xgb_optuna"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
N_TRIALS = 30
N_ESTIMATORS_CEILING = 3000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_cfpb.ubj")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
PREDICTOR_PATH = os.path.join(MODEL_DIR, "cfpb_predictor.joblib")


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

def make_splits(
        features: pd.DataFrame,
        labels: pd.Series,
        test_size:float = TEST_SIZE,
        val_size:float = VAL_SIZE,
        random_state:int = RANDOM_STATE
):
    """
    Stratified spliting data into: train, validation, test.

    Parameters:
    features     : feature DataFrame
    labels       : target Series
    test_size    : size of the test
    val_size     : size of the validation
    random_state : reproducibility seed

    Returns:

    X_tr, X_val, X_train, X_test,
    y_tr, y_val, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size= test_size,
        random_state= random_state,
        stratify= labels
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size= val_size,
        random_state= random_state,
        stratify= y_train
    )

    return (
        X_tr, X_val, X_train, X_test,
        y_tr, y_val, y_train, y_test
    )


def build_encoder(labels: pd.Series) -> LabelEncoder:
    """
    Fit LabelEnoder on all unique target class. 
    """

    encoder = LabelEncoder()
    encoder.fit(labels)

    return encoder

def make_objective(
        X_tr    : pd.DataFrame,
        X_val   : pd.DataFrame,
        y_tr    : pd.Series,
        y_val   : pd.Series,
        encoder : LabelEncoder
):
    """
    Return an Optuna objective bound to given data splits.

    Parameters:
    X_tr    :  feature DataFrame for training
    X_val   :  feature DataFrame for validation
    y_tr    :  label Series for training
    y_val   : label Series for validation
    encoder : fitted LabelEncoder

    Returns:
    objective: Optuna objective
    """

    y_tr_enc = encoder.transform(y_tr)
    y_val_enc = encoder.transform(y_val)

    def objective(trial:optuna.Trial) -> float:
        xgb_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log = True),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-3, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True            
            ),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "max_delta_step": trial.suggest_int(
                "max_delta_step", 0, 10
            ),            
        }

        weights_change = trial.suggest_float("weights_change", 0.1, 0.45)

        with mlflow.start_run(
            nested= True,  run_name=f"trial_{trial.number}"
        ) as child_run:

            mlflow.log_params(
                {**xgb_params, "weights_change": weights_change}
            )

            sample_weights = (
                compute_sample_weight(
                    class_weight= "balanced", y = y_tr
                ) ** weights_change
            )

            model = XGBClassifier(
                **xgb_params,
                n_estimators=N_ESTIMATORS_CEILING, 
                tree_method="hist",
                enable_categorical=True,
                device="cuda",
                eval_metric="mlogloss",
                early_stopping_rounds=30,
                random_state=RANDOM_STATE,
            )

            model.fit(
                X_tr,
                y_tr_enc,
                sample_weight = sample_weights,
                eval_set = [(X_val, y_val_enc)],
                verbose = False
            )

            y_pred = encoder.inverse_transform(
                model.predict(X_val)
                )
            
            f1_macro = f1_score(y_val, y_pred, average="macro")
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("best_iteration", model.best_iteration)            
            mlflow.log_text(
                classification_report(y_val, y_pred),
                "classification_report.txt"
            )
            trial.set_user_attr("run_id", child_run.info.run_id)
            trial.set_user_attr(
                "best_iteration", int(model.best_iteration)
            )
        return f1_macro
        
    return objective

def run_study(
        objective_fn,
        n_trials: int = N_TRIALS,
        storage: str = OPTUNA_STORAGE,
        study_name: str = STUDY_NAME
) -> optuna.Study:
    """
    Create Optuna study in Mlflow paren run.
    
    Parameters:
    objective_fn : callable returned by make_objective
    n_trials     : number of Optuna trials to run

    Returns:
    study  : optuna.Study
    """

    with mlflow.start_run(run_name="study"):
        mlflow.log_param("n_trials", n_trials)
 
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
 
        remaining = n_trials - len(study.trials)
        if remaining > 0:
            study.optimize(objective_fn, n_trials=remaining)
        else:
            print(
                f"Study '{study_name}' already has "
                f"{len(study.trials)} trials — skipping search "
                f"and reusing the existing best result."
            )
 
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_f1_macro_val", study.best_value)
 
    return study

def save_artifacts(
        model: XGBClassifier,
        encoder: LabelEncoder,
        model_path: str = MODEL_PATH,
        encoder_path: str = ENCODER_PATH,
        predictor_path:str = PREDICTOR_PATH
):
    """
    Saving model, encoder and all in one. 

    Parameters:
    model          : trained XGBClassifier
    encoder        : fitted LabelEncoder
    model_path     : local path for XGBoost
    encoder_path   : local path for encoder
    predictor_path : local path for the deployment bundle
    """

    os.makedirs(MODEL_DIR, exist_ok= True)

    model.save_model(model_path)
    joblib.dump(encoder, encoder_path)

    components = {
        "encoder": encoder,
        "model": model
    }

    joblib.dump(components, predictor_path)

    mlflow.log_artifact(model_path, artifact_path="saved_model")
    mlflow.log_artifact(encoder_path, artifact_path="saved_encoder")
    mlflow.log_artifact(predictor_path, artifact_path="saved_predictor")

    print(f"XGBoost model  → {model_path}")
    print(f"Label encoder  → {encoder_path}")
    print(f"Predictor bundle  → {predictor_path}")


def run_final(
    study: optuna.Study,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoder: LabelEncoder,
    random_state: int = RANDOM_STATE
) -> XGBClassifier:
    """
    Retain the best parameters from Optuna study and evaluate on the test set.

    Parameters:
    study : completed Optuna study
    X_train : feature DataFrame for training
    X_test : feature DataFrame for test
    y_train : target Series for training
    y_test : target Series for testing
    encoder :  fitted LabelEncoder
    """
    with mlflow.start_run(run_name="final"):
            
        best_params = study.best_trial.params.copy()
        weights_change = best_params.pop("weights_change")

        best_n_estimators = (
            study.best_trial.user_attrs["best_iteration"] + 1
        )
        mlflow.log_param("n_estimators_used", best_n_estimators)

        sample_weights = (
            compute_sample_weight(
                class_weight="balanced", y=y_train
            ) ** weights_change
        )

        y_train_enc = encoder.transform(y_train)

        final_model = XGBClassifier(
            **best_params,
            n_estimators=best_n_estimators,
            tree_method="hist",
            enable_categorical=True,
            device="cuda",
            eval_metric="mlogloss",
            random_state = random_state
        )

        final_model.fit(X_train, 
                y_train_enc, 
                sample_weight=sample_weights)
        
        y_pred = encoder.inverse_transform(
            final_model.predict(X_test)
        )

        f1_macro = f1_score(y_test, y_pred, average="macro")

        mlflow.log_params(
            {**best_params, "weights_change": weights_change}
        )
        mlflow.log_metric("f1_macro_test", f1_macro)
        mlflow.log_text(
            classification_report(y_test, y_pred),
            "classification_report.txt"
        )
        
        mlflow.xgboost.log_model(
            final_model, artifact_path="model"
        )
        mlflow.set_tag(
            "Training Info",
            "XGBClassifier with Optuna tuning for CFPB data"
        )
        mlflow.set_tag("model_type", "Opt_XGBoost")

        save_artifacts(
            final_model,
            encoder
        )

        return final_model

def main() -> None:
    """Run the full CFPB classification pipeline."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    features, labels = load_data(DATA_URL, TARGET_COLUMN)

    (
    X_tr, X_val, X_train, X_test,
    y_tr, y_val, y_train, y_test
    ) = make_splits(features, labels)

    encoder = build_encoder(y_tr)

    objective_fn = make_objective(
        X_tr, X_val, y_tr, y_val, encoder)
    
    study =  run_study(objective_fn)

    run_final(
        study,
        X_train,
        X_test,
        y_train,
        y_test,
        encoder
    )

if __name__ == "__main__":
    main()












    











