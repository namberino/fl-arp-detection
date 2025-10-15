import warnings

from sklearn.metrics import log_loss
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_ids.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": accuracy}


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
