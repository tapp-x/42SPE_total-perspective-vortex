from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import PowerBandExtractor


def parse_runs(runs_arg):
    if "all" in [r.lower() for r in runs_arg]:
        return list(range(1, 15))
    try:
        return [int(r) for r in runs_arg]
    except ValueError as exc:
        raise ValueError("Invalid run values. Use run numbers or 'all'.") from exc


def build_pipeline():
    return Pipeline(
        [
            ("feature_extraction", PowerBandExtractor()),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear")),
        ]
    )
