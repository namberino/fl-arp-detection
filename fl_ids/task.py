import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from fl_ids.model import DNN


def load_data_non_iid(
    partition_id: int, 
    num_partitions: int, 
    dataset_path: str = "./dataset", 
    num_rows: int = None,
    dirichlet_alpha: float = 0.5,
    min_samples_per_class: int = 2
):
    # Load the dataset and labels
    dataset_df = pd.read_csv(f"{dataset_path}/dataset.csv", index_col=0, nrows=num_rows)
    labels_df = pd.read_csv(f"{dataset_path}/labels.csv", index_col=0, nrows=num_rows)
    
    # Combine dataset and labels
    X = dataset_df.values
    y = labels_df.values.ravel()
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Get unique classes
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Create partitions using Dirichlet allocation
    partition_indices = [[] for _ in range(num_partitions)]
    
    # For each class, distribute samples according to Dirichlet distribution
    for class_id in classes:
        # Get indices of samples belonging to this class
        class_indices = np.where(y == class_id)[0]
        n_class_samples = len(class_indices)
        
        # Sample from Dirichlet distribution to get proportions for each partition
        proportions = np.random.dirichlet(
            alpha=[dirichlet_alpha] * num_partitions
        )
        
        # Calculate number of samples for each partition
        # Reserve min_samples_per_class for each partition first
        reserved_samples = min_samples_per_class * num_partitions
        
        if n_class_samples <= reserved_samples:
            # If not enough samples, distribute evenly
            samples_per_partition = [n_class_samples // num_partitions] * num_partitions
            for i in range(n_class_samples % num_partitions):
                samples_per_partition[i] += 1
        else:
            # Distribute reserved samples first
            remaining_samples = n_class_samples - reserved_samples
            samples_per_partition = [min_samples_per_class] * num_partitions
            
            # Distribute remaining samples according to Dirichlet proportions
            additional_samples = (proportions * remaining_samples).astype(int)
            samples_per_partition = [
                s + a for s, a in zip(samples_per_partition, additional_samples)
            ]
            
            # Handle rounding errors - distribute remaining samples
            diff = n_class_samples - sum(samples_per_partition)
            for i in range(abs(diff)):
                if diff > 0:
                    samples_per_partition[i % num_partitions] += 1
                else:
                    samples_per_partition[i % num_partitions] -= 1
        
        # Shuffle class indices
        np.random.shuffle(class_indices)
        
        # Assign samples to partitions
        start_idx = 0
        for partition_idx, n_samples in enumerate(samples_per_partition):
            end_idx = start_idx + n_samples
            partition_indices[partition_idx].extend(
                class_indices[start_idx:end_idx].tolist()
            )
            start_idx = end_idx
    
    # Get data for this partition
    partition_idx_array = np.array(partition_indices[partition_id])
    
    # Shuffle partition indices
    np.random.shuffle(partition_idx_array)
    
    X_partition = X[partition_idx_array]
    y_partition = y[partition_idx_array]
    
    # Verify all classes are present
    unique_classes_in_partition = np.unique(y_partition)
    if len(unique_classes_in_partition) < n_classes:
        print(f"Warning: Partition {partition_id} missing some classes")
    
    # Split the partition data: 80% train, 20% test
    # Use stratify to maintain class distribution in both sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_partition, y_partition, test_size=0.2, random_state=42, stratify=y_partition
        )
    except ValueError:
        # If stratification fails (too few samples), split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_partition, y_partition, test_size=0.2, random_state=42
        )
    
    # Print partition statistics
    print(f"\nPartition {partition_id} statistics:")
    print(f"  Total samples: {len(y_partition)}")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")
    for class_id in classes:
        n_partition = np.sum(y_partition == class_id)
        n_train = np.sum(y_train == class_id)
        n_test = np.sum(y_test == class_id)
        pct_partition = (n_partition / len(y_partition)) * 100
        print(f"  Class {class_id}: {n_partition:4d} ({pct_partition:5.1f}%) - Train: {n_train:4d}, Test: {n_test:4d}")
    
    return X_train, X_test, y_train, y_test


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


def get_model(n_features: int, n_classes: int, hidden_sizes=[64, 32]):
    model = DNN(n_features, n_classes, hidden_sizes)
    return model


def get_model_params(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_params(model, params):
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    return model
