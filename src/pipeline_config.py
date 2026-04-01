from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from csp import CSPTransformer
from features import PowerBandExtractor


def parse_runs(runs_arg):
    normalized_runs = []
    for value in runs_arg:
        normalized_runs.extend(value.split())

    if "all" in [r.lower() for r in normalized_runs]:
        return list(range(1, 15))
    try:
        return [int(r) for r in normalized_runs]
    except ValueError as exc:
        raise ValueError("Invalid run values. Use run numbers or 'all'.") from exc


def pipeline_suffix(dim_red, n_components):
    if dim_red == "pca":
        return f"pca{n_components}"
    if dim_red == "csp":
        return f"csp{n_components}"
    return "base"


def build_pipeline(dim_red="csp", n_components=5):
    if dim_red == "csp":
        steps = [
            ("dimensionality_reduction", CSPTransformer(n_components=n_components)),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear")),
        ]
        return Pipeline(steps)

    steps = [
        ("feature_extraction", PowerBandExtractor()),
        ("scaler", StandardScaler()),
    ]

    if dim_red == "pca":
        steps.append(("dimensionality_reduction", PCA(n_components=n_components)))

    steps.append(("classifier", SVC(kernel="linear")))
    return Pipeline(steps)
