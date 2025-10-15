import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(partition_id: int, num_partitions: int, dataset_path: str = "./dataset", num_rows: int = None):
    # Load the dataset and labels
    dataset_df = pd.read_csv(f"{dataset_path}/dataset.csv", index_col=0, nrows=num_rows)
    labels_df = pd.read_csv(f"{dataset_path}/labels.csv", index_col=0, nrows=num_rows)
    
    # if num_rows is not None and num_rows < len(dataset_df):
    #     dataset_df = dataset_df.iloc[:num_rows]
    #     labels_df = labels_df.iloc[:num_rows]
    
    # Combine dataset and labels
    X = dataset_df.values
    y = labels_df.values.ravel()
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Calculate partition size
    total_samples = len(X)
    partition_size = total_samples // num_partitions
    
    # Get data for this partition
    start_idx = partition_id * partition_size
    if partition_id == num_partitions - 1:
        # Last partition gets remaining data
        end_idx = total_samples
    else:
        end_idx = start_idx + partition_size
    
    X_partition = X[start_idx:end_idx]
    y_partition = y[start_idx:end_idx]
    
    # Split the partition data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=42, stratify=y_partition
    )
    
    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int):
    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
        random_state=42,
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


def set_initial_params(model, n_features: int, n_classes: int):
    model.classes_ = np.array([i for i in range(n_classes)])
    
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
