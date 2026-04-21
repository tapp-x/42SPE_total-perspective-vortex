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
        raise ValueError("Invalid run values. Use run numbers or all.") from exc


def parse_subjects(subject_args, max_subject=109):
    normalized_subjects = []
    for value in subject_args:
        normalized_subjects.extend(value.split())

    if "all" in [value.lower() for value in normalized_subjects]:
        return list(range(1, max_subject + 1))
    try:
        return [int(value) for value in normalized_subjects]
    except ValueError as exc:
        raise ValueError("Invalid subject values. Use subject numbers or all.") from exc


def pipeline_suffix(dim_red, n_components):
    if dim_red == "pca":
        return f"pca{n_components}"
    if dim_red == "csp":
        return f"csp{n_components}"
    return "base"


def default_model_path(subject, runs, dim_red, n_components):
    runs_slug = "all" if runs == list(range(1, 15)) else "-".join(f"{r:02d}" for r in runs)
    variant_slug = pipeline_suffix(dim_red, n_components)
    return f"models/s{subject:03d}_runs_{runs_slug}_{variant_slug}.joblib"


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
