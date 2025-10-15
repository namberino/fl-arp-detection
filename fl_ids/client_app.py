import warnings
import numpy as np

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_ids.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)


def _safe_metrics(y_true, y_pred):
    """
    Compute accuracy, f1, recall, precision safely.
    Returns dict with numeric values (float).
    Uses 'weighted' average for multiclass.
    """
    if y_true is None or len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "f1": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
        }

    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = float("nan")

    try:
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        f1 = float("nan")

    try:
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        rec = float("nan")

    try:
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        prec = float("nan")

    return {"accuracy": acc, "f1": f1, "recall": rec, "precision": prec}


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        # Load parameters into local model
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit on local training data if available
            if self.X_train is not None and len(self.X_train) > 0:
                self.model.fit(self.X_train, self.y_train)

        # Compute training metrics (if training data present)
        if self.X_train is not None and len(self.X_train) > 0:
            try:
                y_pred_train = self.model.predict(self.X_train)
                train_metrics = _safe_metrics(self.y_train, y_pred_train)
            except Exception:
                train_metrics = {
                    "train_accuracy": float("nan"),
                    "train_f1": float("nan"),
                    "train_recall": float("nan"),
                    "train_precision": float("nan"),
                }
        else:
            train_metrics = {
                "train_accuracy": float("nan"),
                "train_f1": float("nan"),
                "train_recall": float("nan"),
                "train_precision": float("nan"),
            }

        # prefix keys with 'train_' to make it explicit in server summary
        metrics_prefixed = {
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_recall": train_metrics["recall"],
            "train_precision": train_metrics["precision"],
        }

        return get_model_params(self.model), len(self.X_train), metrics_prefixed

    def evaluate(self, parameters, config):
        # Load parameters into local model
        set_model_params(self.model, parameters)

        # If there is no test data, return nan loss and metrics
        if self.X_test is None or len(self.X_test) == 0:
            loss = float("nan")
            metrics = {
                "accuracy": float("nan"),
                "f1": float("nan"),
                "recall": float("nan"),
                "precision": float("nan"),
            }
            return loss, 0, metrics

        # Compute loss (use predict_proba when available)
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(self.X_test)
                loss = float(log_loss(self.y_test, proba))
            else:
                # fallback: use predicted labels to compute 0/1 loss (not log_loss)
                y_pred = self.model.predict(self.X_test)
                # compute "loss" as 1 - accuracy when predict_proba unavailable
                loss = float(1.0 - accuracy_score(self.y_test, y_pred))
        except Exception:
            loss = float("nan")

        # compute other metrics
        try:
            y_pred = self.model.predict(self.X_test)
            eval_metrics = _safe_metrics(self.y_test, y_pred)
        except Exception:
            eval_metrics = {
                "accuracy": float("nan"),
                "f1": float("nan"),
                "recall": float("nan"),
                "precision": float("nan"),
            }

        # Include accuracy/f1/recall/precision in metrics dict
        metrics = {
            "accuracy": eval_metrics["accuracy"],
            "f1": eval_metrics["f1"],
            "recall": eval_metrics["recall"],
            "precision": eval_metrics["precision"],
        }

        return loss, len(self.X_test), metrics


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read dataset configuration from run_config
    dataset_path = context.run_config.get("dataset-path", "./dataset")
    dataset_rows = int(context.run_config.get("dataset-rows", -1))
    dataset_shuffle = bool(context.run_config.get("dataset-shuffle", True))

    X_train, X_test, y_train, y_test = load_data(
        partition_id, num_partitions, dataset_path, dataset_rows, dataset_shuffle
    )

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Set initial parameters using actual feature/class sizes
    n_features = X_train.shape[1] if X_train is not None and len(X_train) > 0 else None
    n_classes = int(len(np.unique(y_train))) if y_train is not None and len(y_train) > 0 else None
    set_initial_params(model, n_features=n_features, n_classes=n_classes)

    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
