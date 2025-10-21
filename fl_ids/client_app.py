import warnings
import torch

from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_ids.task import (
    get_model,
    get_model_params,
    load_data,
    load_data_non_iid,
    set_model_params,
)
from fl_ids.adversarial import AdversarialTrainer


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, adv_config, local_epochs, batch_size, learning_rate, device):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.model.to(device)
        
        # Initialize adversarial trainer
        self.adversarial_trainer = AdversarialTrainer(
            model=self.model,
            attack_type=adv_config["attack_type"],
            epsilon=adv_config["epsilon"],
            alpha=adv_config["alpha"],
            num_iter=adv_config["num_iter"],
            random_start=adv_config["random_start"],
            device=device,
        )

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Train with adversarial examples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.adversarial_trainer.fit_with_adversarial(
                self.X_train, 
                self.y_train,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate
            )

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.eval()
        
        with torch.no_grad():
            # Convert test data to tensors
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_test_tensor = torch.LongTensor(self.y_test).to(self.device)
            
            # Get predictions
            outputs = self.model(X_test_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            y_pred = predicted.cpu().numpy()
        
        # Calculate metrics
        loss = log_loss(self.y_test, probabilities)
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
    
    use_non_iid = context.run_config.get("use-non-iid", False)
    
    if use_non_iid:
        dirichlet_alpha = context.run_config.get("dirichlet-alpha", 0.5)
        min_samples_per_class = context.run_config.get("min-samples-per-class", 2)
        
        X_train, X_test, y_train, y_test = load_data_non_iid(
            partition_id, 
            num_partitions, 
            dataset_path, 
            num_rows,
            dirichlet_alpha=dirichlet_alpha,
            min_samples_per_class=min_samples_per_class
        )
    else:
        X_train, X_test, y_train, y_test = load_data(
            partition_id, num_partitions, dataset_path, num_rows
        )

    # Get dataset dimensions for model initialization
    n_features = X_train.shape[1]
    n_classes = len(set(y_train))

    # Training hyperparameters
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config.get("batch-size", 32)
    learning_rate = context.run_config.get("learning-rate", 0.001)
    hidden_sizes = tuple(int(i) for i in context.run_config.get("hidden-sizes", "128,64").split(","))
    device = context.run_config.get("device", "cpu")

    # Create DNN Model
    model = get_model(n_features, n_classes, hidden_sizes)

    # Adversarial training configuration
    adv_config = {
        "attack_type": context.run_config.get("adversarial-attack-type", "none"),
        "epsilon": context.run_config.get("adversarial-epsilon", 0.1),
        "alpha": context.run_config.get("adversarial-alpha", 0.01),
        "num_iter": context.run_config.get("adversarial-num-iter", 10),
        "random_start": context.run_config.get("adversarial-random-start", True),
    }

    return FlowerClient(
        model, X_train, X_test, y_train, y_test, 
        adv_config, local_epochs, batch_size, learning_rate, device
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
