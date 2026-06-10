import mlflow
import mlflow.sklearn
import joblib
import logging
import os
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.features import build_flat_feature_table, get_feature_cols

logger = logging.getLogger(__name__)

def train():
    os.makedirs("model", exist_ok=True)
    os.makedirs("monitoring", exist_ok=True)

    logger.info("Loading and engineering features...")
    df           = build_flat_feature_table()
    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df["TARGET"]

    joblib.dump(feature_cols, "model/feature_cols.pkl")
    logger.info(f"Using {len(feature_cols)} features")

    cutoff = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    X_train.to_parquet("monitoring/reference_data.parquet", index=False)

    neg, pos         = (y_train==0).sum(), (y_train==1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"Default rate: {pos/(pos+neg):.1%} | scale_pos_weight: {scale_pos_weight:.1f}")

    mlflow.set_experiment("bnpl-credit-risk")

    with mlflow.start_run():
        lgbm = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, n_jobs=-1,
            random_state=42, verbose=-1,
        )

        logger.info("Running cross-validation (5 folds)...")
        cv = cross_validate(
            lgbm, X_train, y_train,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring=["roc_auc", "average_precision"],
        )
        logger.info(f"CV AUC: {cv['test_roc_auc'].mean():.4f}")
        logger.info(f"CV AP:  {cv['test_average_precision'].mean():.4f}")

        logger.info("Training final model...")
        lgbm.fit(X_train, y_train)

        logger.info("Calibrating probabilities...")
        calibrated = CalibratedClassifierCV(lgbm, cv=None, method="isotonic")
        calibrated.fit(X_test, y_test)

        y_prob = calibrated.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        ap     = average_precision_score(y_test, y_prob)
        gini   = 2 * auc - 1
        logger.info(f"Holdout AUC: {auc:.4f} | Gini: {gini:.4f} | AP: {ap:.4f}")

        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        f1s       = 2*prec*rec / (prec+rec+1e-8)
        threshold = float(thresholds[np.argmax(f1s)])
        logger.info(f"Optimal threshold: {threshold:.4f}")

        logger.info("Computing SHAP values (takes ~1 minute)...")
        explainer   = shap.TreeExplainer(lgbm)
        sample      = X_test.sample(500, random_state=42)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        plt.savefig("monitoring/shap_summary.png", bbox_inches="tight", dpi=120)
        plt.close()
        logger.info("SHAP plot saved to monitoring/shap_summary.png")

        mlflow.log_metrics({"cv_auc": cv["test_roc_auc"].mean(),
                            "holdout_auc": auc, "gini": gini, "ap": ap})
        mlflow.sklearn.log_model(calibrated, "model")

        joblib.dump(calibrated, "model/lgbm_calibrated.pkl")
        joblib.dump(explainer,  "model/shap_explainer.pkl")
        joblib.dump(threshold,  "model/threshold.pkl")
        logger.info("All model files saved to model/")

if __name__ == "__main__":
    from utils.logging_config import setup_logging
    setup_logging()
    train()