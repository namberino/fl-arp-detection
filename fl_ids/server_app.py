import pandas as pd

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fl_ids.task import get_model, get_model_params, set_initial_params


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    dataset_path = context.run_config["dataset-path"]
    num_rows = context.run_config.get("num-rows", None)

    # Load dataset to get dimensions
    dataset_df = pd.read_csv(f"{dataset_path}/dataset.csv", index_col=0)
    labels_df = pd.read_csv(f"{dataset_path}/labels.csv", index_col=0)
    
    # Limit rows if specified
    if num_rows is not None and num_rows < len(dataset_df):
        dataset_df = dataset_df.iloc[:num_rows]
        labels_df = labels_df.iloc[:num_rows]
    
    n_features = dataset_df.shape[1]
    n_classes = len(labels_df.iloc[:, 0].unique())

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters
    set_initial_params(model, n_features, n_classes)

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
