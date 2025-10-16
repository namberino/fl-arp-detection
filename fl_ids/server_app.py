import pandas as pd
from typing import List, Tuple, Dict, Optional
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from fl_ids.task import get_model, get_model_params


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Get total number of examples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Initialize aggregated metrics
    aggregated = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "clean_accuracy": 0.0,
        "adversarial_accuracy": 0.0,
        "robustness_gap": 0.0,
    }
    
    # Weighted average of each metric
    for num_examples, m in metrics:
        weight = num_examples / total_examples
        aggregated["accuracy"] += m.get("accuracy", 0) * weight
        aggregated["precision"] += m.get("precision", 0) * weight
        aggregated["recall"] += m.get("recall", 0) * weight
        aggregated["f1"] += m.get("f1", 0) * weight
        aggregated["clean_accuracy"] += m.get("clean_accuracy", 0) * weight
        aggregated["adversarial_accuracy"] += m.get("adversarial_accuracy", 0) * weight
        aggregated["robustness_gap"] += m.get("robustness_gap", 0) * weight
    
    return aggregated


class CustomFedAvg(FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Dict]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:        
        if not results:
            return None, {}
        
        # Call parent class to aggregate
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Print metrics for this round
        if metrics_aggregated:
            print(f"\n{'='*70}")
            print(f"Round {server_round} - Global Model Performance")
            print(f"{'='*70}")
            print(f"Loss:                    {loss_aggregated:.4f}")
            print(f"Accuracy:                {metrics_aggregated.get('accuracy', 0):.4f}")
            print(f"Precision:               {metrics_aggregated.get('precision', 0):.4f}")
            print(f"Recall:                  {metrics_aggregated.get('recall', 0):.4f}")
            print(f"F1 Score:                {metrics_aggregated.get('f1', 0):.4f}")
            
            # Print adversarial robustness metrics if available
            if metrics_aggregated.get('adversarial_accuracy', 0) > 0:
                print(f"\n--- Adversarial Robustness ---")
                print(f"Clean Accuracy:          {metrics_aggregated.get('clean_accuracy', 0):.4f}")
                print(f"Adversarial Accuracy:    {metrics_aggregated.get('adversarial_accuracy', 0):.4f}")
                print(f"Robustness Gap:          {metrics_aggregated.get('robustness_gap', 0):.4f}")
            
            print(f"{'='*70}\n")
        
        return loss_aggregated, metrics_aggregated


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    dataset_path = context.run_config["dataset-path"]
    num_rows = context.run_config.get("num-rows", None)

    # Load dataset to get dimensions
    dataset_df = pd.read_csv(f"{dataset_path}/dataset.csv", index_col=0, nrows=num_rows)
    labels_df = pd.read_csv(f"{dataset_path}/labels.csv", index_col=0, nrows=num_rows)
    
    n_features = dataset_df.shape[1]
    n_classes = len(labels_df.iloc[:, 0].unique())

    # Get model configuration
    hidden_sizes = tuple(int(i) for i in context.run_config.get("hidden-sizes", "128,64").split(","))

    # Create DNN Model
    model = get_model(n_features, n_classes, hidden_sizes)

    # Get initial parameters
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define custom strategy with metric aggregation
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
