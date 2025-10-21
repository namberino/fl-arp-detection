import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fl_ids.task import load_data_non_iid, load_data
import argparse


def visualize_distribution(
    num_partitions: int = 5,
    dataset_path: str = "./dataset",
    num_rows: int = 50000,
    dirichlet_alpha: float = 0.5,
    min_samples_per_class: int = 2,
    compare_iid: bool = True,
    save_path: str = None
):
    print(f"Loading data for {num_partitions} partitions...")
    print(f"Dirichlet Alpha: {dirichlet_alpha}")
    print(f"Min samples per class: {min_samples_per_class}\n")
    
    # Collect data for all partitions
    non_iid_distributions = []
    iid_distributions = []
    
    # Load non-IID data
    print("Loading Non-IID distribution...")
    for partition_id in range(num_partitions):
        X_train, X_test, y_train, y_test = load_data_non_iid(
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset_path=dataset_path,
            num_rows=num_rows,
            dirichlet_alpha=dirichlet_alpha,
            min_samples_per_class=min_samples_per_class
        )
        
        # Combine train and test for distribution visualization
        y_combined = np.concatenate([y_train, y_test])
        classes = np.unique(y_combined)
        
        # Count samples per class
        class_counts = {}
        for class_id in classes:
            class_counts[class_id] = np.sum(y_combined == class_id)
        
        non_iid_distributions.append(class_counts)
    
    # Load IID data for comparison if requested
    if compare_iid:
        print("\nLoading IID distribution for comparison...")
        for partition_id in range(num_partitions):
            X_train, X_test, y_train, y_test = load_data(
                partition_id=partition_id,
                num_partitions=num_partitions,
                dataset_path=dataset_path,
                num_rows=num_rows
            )
            
            y_combined = np.concatenate([y_train, y_test])
            classes = np.unique(y_combined)
            
            class_counts = {}
            for class_id in classes:
                class_counts[class_id] = np.sum(y_combined == class_id)
            
            iid_distributions.append(class_counts)
    
    # Create visualizations
    n_classes = len(classes)
    
    if compare_iid:
        fig1 = plt.figure(figsize=(15, 8))
        fig2 = plt.figure(figsize=(15, 8))
        gs1 = fig1.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        gs2 = fig2.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    else:
        fig1 = plt.figure(figsize=(16, 10))
        gs1 = fig1.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Non-IID Bar Chart
    ax1 = fig1.add_subplot(gs1[1, 0])
    plot_stacked_bar(ax1, non_iid_distributions, num_partitions, n_classes, 
                     f"Non-IID distribution (α={dirichlet_alpha})")
    
    # Non-IID Heatmap
    ax3 = fig1.add_subplot(gs1[1, 1])
    plot_heatmap(ax3, non_iid_distributions, num_partitions, n_classes,
                f"Non-IID distribution heatmap (α={dirichlet_alpha})")
    
    # IID comparison
    if compare_iid:
        ax5 = fig2.add_subplot(gs1[1, 0])
        plot_stacked_bar(ax5, iid_distributions, num_partitions, n_classes,
                        "IID distribution")
        
        ax6 = fig2.add_subplot(gs1[1, 1])
        plot_heatmap(ax6, iid_distributions, num_partitions, n_classes,
                    "IID distribution heatmap")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print_summary_statistics(non_iid_distributions, num_partitions, n_classes, "Non-IID")
    if compare_iid:
        print_summary_statistics(iid_distributions, num_partitions, n_classes, "IID")


def plot_stacked_bar(ax, distributions, num_partitions, n_classes, title):
    partitions = [f"Device {i}" for i in range(num_partitions)]
    
    # Prepare data for stacking
    data_by_class = {}
    for class_id in range(n_classes):
        data_by_class[class_id] = [dist.get(class_id, 0) for dist in distributions]
    
    # Create stacked bars
    bottom = np.zeros(num_partitions)
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for class_id in range(n_classes):
        ax.bar(partitions, data_by_class[class_id], bottom=bottom, 
               label=f"Class {class_id}", color=colors[class_id], alpha=0.8)
        bottom += data_by_class[class_id]
    
    ax.set_xlabel("Partition")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_percentage_bar(ax, distributions, num_partitions, n_classes, title):
    partitions = [f"Device {i}" for i in range(num_partitions)]
    
    # Calculate percentages
    totals = [sum(dist.values()) for dist in distributions]
    data_by_class = {}
    
    for class_id in range(n_classes):
        data_by_class[class_id] = [
            (dist.get(class_id, 0) / totals[i] * 100) if totals[i] > 0 else 0
            for i, dist in enumerate(distributions)
        ]
    
    # Create stacked bars
    bottom = np.zeros(num_partitions)
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for class_id in range(n_classes):
        ax.bar(partitions, data_by_class[class_id], bottom=bottom,
               label=f"Class {class_id}", color=colors[class_id], alpha=0.8)
        
        # Add percentage labels
        for i, pct in enumerate(data_by_class[class_id]):
            if pct > 5:  # Only label if > 5%
                ax.text(i, bottom[i] + pct/2, f"{pct:.1f}%", 
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        bottom += data_by_class[class_id]
    
    ax.set_xlabel("Partition", fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_heatmap(ax, distributions, num_partitions, n_classes, title):
    # Create matrix for heatmap
    matrix = np.zeros((n_classes, num_partitions))
    
    for partition_id, dist in enumerate(distributions):
        for class_id in range(n_classes):
            matrix[class_id, partition_id] = dist.get(class_id, 0)
    
    # Plot heatmap
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                xticklabels=[f"Device {i}" for i in range(num_partitions)],
                yticklabels=[f"Class {i}" for i in range(n_classes)],
                cbar_kws={'label': 'Sample Count'}, ax=ax)
    
    ax.set_xlabel("Partition")
    # ax.set_ylabel("Class")
    ax.set_title(title)


def plot_distribution_stats(ax, distributions, num_partitions, n_classes, title):
    ax.axis('off')
    
    # Calculate statistics
    totals = [sum(dist.values()) for dist in distributions]
    
    # Calculate class imbalance per partition (std of percentages)
    imbalances = []
    for dist in distributions:
        total = sum(dist.values())
        if total > 0:
            percentages = [dist.get(i, 0) / total * 100 for i in range(n_classes)]
            imbalance = np.std(percentages)
            imbalances.append(imbalance)
    
    # Calculate overall heterogeneity (how different partitions are from each other)
    class_distributions = []
    for class_id in range(n_classes):
        class_dist = [dist.get(class_id, 0) for dist in distributions]
        class_distributions.append(class_dist)
    
    # Coefficient of variation for each class across partitions
    cvs = []
    for class_dist in class_distributions:
        mean = np.mean(class_dist)
        if mean > 0:
            cv = np.std(class_dist) / mean
            cvs.append(cv)
    
    # Create statistics text
    stats_text = f"""
    DISTRIBUTION STATISTICS
    {'='*40}
    
    Partition Sizes:
    {'  Min: ':>12} {min(totals):>6.0f} samples
    {'  Max: ':>12} {max(totals):>6.0f} samples
    {'  Mean: ':>12} {np.mean(totals):>6.1f} samples
    {'  Std: ':>12} {np.std(totals):>6.1f} samples
    
    Class Imbalance per Partition:
    {'  Mean: ':>12} {np.mean(imbalances):>6.2f}% std
    {'  Max: ':>12} {np.max(imbalances):>6.2f}% std
    
    Heterogeneity Across Partitions:
    {'  Mean CV: ':>12} {np.mean(cvs):>6.3f}
    {'  Max CV: ':>12} {np.max(cvs):>6.3f}
    
    Class Distribution per Partition:
    """
    
    for partition_id, dist in enumerate(distributions):
        total = sum(dist.values())
        percentages = [f"{dist.get(i, 0)/total*100:5.1f}%" for i in range(n_classes)]
        stats_text += f"\n    Device {partition_id}: {' | '.join(percentages)}"
    
    stats_text += f"""
    
    {'='*40}
    Interpretation:
    - Higher CV = More heterogeneous
    - Higher imbalance = More skewed
    - Lower values = More IID-like
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def print_summary_statistics(distributions, num_partitions, n_classes, label):
    print(f"\n{'='*60}")
    print(f"{label} Distribution Summary")
    print(f"{'='*60}")
    
    for partition_id, dist in enumerate(distributions):
        total = sum(dist.values())
        print(f"\nPartition {partition_id}: {total} total samples")
        for class_id in range(n_classes):
            count = dist.get(class_id, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Class {class_id}: {count:4d} samples ({pct:5.1f}%)")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize data distribution across federated learning partitions"
    )
    parser.add_argument(
        "--num-partitions", 
        type=int, 
        default=5,
        help="Number of partitions/devices (default: 5)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./dataset",
        help="Path to dataset directory (default: ./dataset)"
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=50000,
        help="Number of rows to load (default: 3000)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha parameter (default: 0.5)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples per class per partition (default: 2)"
    )
    parser.add_argument(
        "--no-compare-iid",
        action="store_true",
        help="Don't compare with IID distribution"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the figure (default: None, just display)"
    )
    
    args = parser.parse_args()
    
    visualize_distribution(
        num_partitions=args.num_partitions,
        dataset_path=args.dataset_path,
        num_rows=args.num_rows,
        dirichlet_alpha=args.alpha,
        min_samples_per_class=args.min_samples,
        compare_iid=not args.no_compare_iid,
        save_path=args.save
    )


if __name__ == "__main__":
    main()
