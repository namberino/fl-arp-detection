from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fl_ids.task import get_model, get_model_params, set_initial_params, _read_csvs
from pathlib import Path
import numpy as np

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Try to infer n_features and n_classes from dataset if path is provided
    dataset_path = Path(context.run_config.get("dataset-path", "./dataset"))
    dataset_rows = int(context.run_config.get("dataset-rows", -1))

    n_features = None
    n_classes = None
    try:
        if dataset_path.exists():
            X, y = _read_csvs(dataset_path)
            if dataset_rows and dataset_rows > 0:
                X = X[:dataset_rows]
                y = y[:dataset_rows]
            n_features = X.shape[1]
            n_classes = int(len(np.unique(y)))
    except Exception:
        # If inference fails, allow config overrides (or fall back to defaults in set_initial_params)
        n_features = context.run_config.get("dataset-n-features", None)
        n_classes = context.run_config.get("dataset-n-classes", None)

    # Setting initial parameters with inferred or configured sizes
    set_initial_params(model, n_features=n_features, n_classes=n_classes)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
