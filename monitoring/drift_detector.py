import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logger = logging.getLogger(__name__)

def check_drift(current_data, reference_path="monitoring/reference_data.parquet",
                threshold=0.25):
    reference = pd.read_parquet(reference_path)
    shared    = [c for c in reference.columns if c in current_data.columns]

    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=reference[shared], current_data=current_data[shared])
    report.save_html("monitoring/drift_report.html")

    result      = report.as_dict()
    drift_info  = result["metrics"][1]["result"]
    drift_share = drift_info["share_of_drifted_columns"]
    n_drifted   = drift_info["number_of_drifted_columns"]

    if drift_share >= threshold:
        logger.warning(f"Drift detected: {n_drifted}/{len(shared)} features shifted")
    else:
        logger.info(f"No significant drift. Share: {drift_share:.0%}")

    return {
        "n_features_total":   len(shared),
        "n_features_drifted": n_drifted,
        "drift_share":        round(drift_share, 3),
        "dataset_drift":      drift_info["dataset_drift"],
        "should_retrain":     drift_share >= threshold,
    }
