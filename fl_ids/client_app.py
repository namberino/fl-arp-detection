import warnings
import pandas as pd

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_ids.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)
from fl_ids.adversarial import AdversarialTrainer


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, adv_config):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize adversarial trainer
        self.adversarial_trainer = AdversarialTrainer(
            model=self.model,
            attack_type=adv_config["attack_type"],
            epsilon=adv_config["epsilon"],
            alpha=adv_config["alpha"],
            num_iter=adv_config["num_iter"],
            random_start=adv_config["random_start"],
        )

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Train with adversarial examples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.adversarial_trainer.fit_with_adversarial(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        # Standard evaluation metrics
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

        # Evaluate robustness against adversarial examples
        robustness_metrics = self.adversarial_trainer.evaluate_robustness(
            self.X_test, self.y_test
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "clean_accuracy": robustness_metrics["clean_accuracy"],
            "adversarial_accuracy": robustness_metrics["adversarial_accuracy"],
            "robustness_gap": robustness_metrics["robustness_gap"],
        }

        return loss, len(self.X_test), metrics


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load data with configuration
    dataset_path = context.run_config["dataset-path"]
    num_rows = context.run_config.get("num-rows", None)
    
    X_train, X_test, y_train, y_test = load_data(
        partition_id, num_partitions, dataset_path, num_rows
    )

    # Get dataset dimensions for model initialization
    n_features = X_train.shape[1]
    n_classes = len(set(y_train))

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters
    set_initial_params(model, n_features, n_classes)

    # Adversarial training configuration
    adv_config = {
        "attack_type": context.run_config.get("adversarial-attack-type", "none"),
        "epsilon": context.run_config.get("adversarial-epsilon", 0.1),
        "alpha": context.run_config.get("adversarial-alpha", 0.01),
        "num_iter": context.run_config.get("adversarial-num-iter", 10),
        "random_start": context.run_config.get("adversarial-random-start", True),
    }

    return FlowerClient(model, X_train, X_test, y_train, y_test, adv_config).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
