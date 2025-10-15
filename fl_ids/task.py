import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Cache loaded dataset across calls
_cached_X = None
_cached_y = None
_cached_rows_loaded = None

def _read_csvs(dataset_path: Path):
    """Read dataset.csv and labels.csv (drop the first index column)."""
    ds_file = dataset_path / "dataset.csv"
    labels_file = dataset_path / "labels.csv"

    if not ds_file.exists():
        raise FileNotFoundError(f"dataset.csv not found at {ds_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_file}")

    # read and drop first column (index), if present
    df_X = pd.read_csv(ds_file, index_col=0)
    df_y = pd.read_csv(labels_file, index_col=0)

    # If labels file has a single column, flatten it
    if df_y.shape[1] == 1:
        y = df_y.iloc[:, 0].to_numpy()
    else:
        # if multiple columns, try to find a sensible label column
        y = df_y.iloc[:, 0].to_numpy()

    X = df_X.to_numpy()
    return X, y


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_path: str = "./dataset",
    dataset_rows: int = -1,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Load a partition of the CSV dataset.

    - dataset_path: path containing dataset.csv and labels.csv
    - dataset_rows: total number of rows to consider from the dataset (-1 = all)
    - partitioning: IID split by splitting indices with np.array_split
    """
    global _cached_X, _cached_y, _cached_rows_loaded

    dataset_path = Path(dataset_path)

    if _cached_X is None or _cached_rows_loaded != dataset_rows:
        X, y = _read_csvs(dataset_path)

        if dataset_rows is not None and dataset_rows > 0:
            if dataset_rows > len(X):
                raise ValueError(
                    f"dataset_rows ({dataset_rows}) is greater than available rows ({len(X)})"
                )
            X = X[:dataset_rows]
            y = y[:dataset_rows]

        _cached_X = X
        _cached_y = y
        _cached_rows_loaded = dataset_rows

    X_all = _cached_X
    y_all = _cached_y

    n_rows = len(X_all)
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_rows)
        X_all = X_all[perm]
        y_all = y_all[perm]

    # Partition indices IID across clients
    all_indices = np.arange(len(X_all))
    splits = np.array_split(all_indices, num_partitions)
    if partition_id < 0 or partition_id >= num_partitions:
        raise IndexError(f"partition_id {partition_id} out of range [0, {num_partitions})")
    part_idx = splits[partition_id]

    X_part = X_all[part_idx]
    y_part = y_all[part_idx]

    # Split the local partition into 80% train / 20% test
    split_at = int(0.8 * len(X_part)) if len(X_part) > 1 else len(X_part)
    X_train, X_test = X_part[:split_at], X_part[split_at:]
    y_train, y_test = y_part[:split_at], y_part[split_at:]

    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int):
    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model, n_features: int = None, n_classes: int = None):
    """
    Initialize model.classes_, coef_ and intercept_ to zeros with shapes derived
    from n_features and n_classes. If not provided, fall back to MNIST defaults.
    """
    # Fallbacks (previous MNIST defaults)
    if n_classes is None:
        n_classes = 10
    if n_features is None:
        n_features = 784

    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
