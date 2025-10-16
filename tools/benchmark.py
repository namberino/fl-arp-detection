# centralized_benchmark.py
import argparse
import json
import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from fl_ids.task import load_data, get_model
from fl_ids.adversarial import AdversarialTrainer

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_tensor = torch.LongTensor(y_test).to(device)

        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, preds = torch.max(outputs, 1)
        y_pred = preds.cpu().numpy()

    # log-loss requires probability predictions (probs) and true labels
    loss = log_loss(y_test, probs, labels=np.unique(y_test))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def run_centralized(
    dataset_path,
    num_rows,
    hidden_sizes,
    device,
    epochs,
    batch_size,
    learning_rate,
    attack_type,
    epsilon,
    alpha,
    num_iter,
    random_start,
    output_dir,
    seed=42,
):
    set_seed(seed)
    device = torch.device(device)

    print("Loading full dataset (partition_id=0, num_partitions=1) ...")
    X_train, X_test, y_train, y_test = load_data(0, 1, dataset_path, num_rows)

    n_features = X_train.shape[1]
    n_classes = len(set(y_train))
    print(f"Dataset: n_features={n_features}, n_classes={n_classes}")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Create baseline model (standard training)
    model_baseline = get_model(n_features, n_classes, hidden_sizes)
    model_baseline.to(device)
    trainer_baseline = AdversarialTrainer(
        model=model_baseline,
        attack_type="none",  # no adversarial training
        epsilon=0.0,
        alpha=0.0,
        num_iter=0,
        random_start=False,
        device=str(device),
    )

    # Create adversarial model (PGD/FGSM based on attack_type)
    model_adv = get_model(n_features, n_classes, hidden_sizes)
    model_adv.to(device)
    trainer_adv = AdversarialTrainer(
        model=model_adv,
        attack_type=attack_type,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        random_start=random_start,
        device=str(device),
    )

    # Train baseline
    print("\n=== Training baseline (standard) centralized model ===")
    trainer_baseline.fit_with_adversarial(
        X_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    baseline_metrics = evaluate_model(model_baseline, X_test, y_test, device)
    # Baseline robustness (since attack_type==none, adversarial accuracy == clean)
    robustness_baseline = trainer_baseline.evaluate_robustness(X_test, y_test)

    # Train adversarial
    print("\n=== Training adversarial centralized model (attack_type=%s) ===" % attack_type)
    trainer_adv.fit_with_adversarial(
        X_train, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    adv_metrics = evaluate_model(model_adv, X_test, y_test, device)
    robustness_adv = trainer_adv.evaluate_robustness(X_test, y_test)

    # Aggregate outputs
    results = {
        "baseline": {
            "metrics": baseline_metrics,
            "robustness": robustness_baseline,
        },
        "adversarial": {
            "config": {
                "attack_type": attack_type,
                "epsilon": epsilon,
                "alpha": alpha,
                "num_iter": num_iter,
                "random_start": random_start,
            },
            "metrics": adv_metrics,
            "robustness": robustness_adv,
        },
    }

    # Save models and results
    os.makedirs(output_dir, exist_ok=True)
    baseline_path = os.path.join(output_dir, "centralized_baseline.pt")
    adv_path = os.path.join(output_dir, "centralized_adversarial.pt")
    torch.save(model_baseline.state_dict(), baseline_path)
    torch.save(model_adv.state_dict(), adv_path)

    results_path = os.path.join(output_dir, "centralized_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    def print_block(title, m, r=None):
        print("\n" + ("-" * 60))
        print(title)
        print("-" * 60)
        print(f"Loss:      {m['loss']:.4f}")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1 score:  {m['f1']:.4f}")
        if r is not None:
            print("\nRobustness:")
            print(f"  Clean Acc:       {r['clean_accuracy']:.4f}")
            print(f"  Adversarial Acc: {r['adversarial_accuracy']:.4f}")
            print(f"  Robustness Gap:  {r['robustness_gap']:.4f}")

    print_block("BASELINE (centralized)", results["baseline"]["metrics"], results["baseline"]["robustness"])
    print_block("ADVERSARIAL (centralized)", results["adversarial"]["metrics"], results["adversarial"]["robustness"])

    print(f"\nModels saved to: {output_dir}")
    print(f"Results saved to: {results_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized benchmark for IDS (baseline vs adversarial)")
    parser.add_argument("--dataset-path", type=str, default="./dataset", help="path to dataset folder containing dataset.csv and labels.csv")
    parser.add_argument("--num-rows", type=int, default=50000, help="limit number of rows read from dataset (optional)")
    parser.add_argument("--hidden-sizes", type=str, default="128,64", help="comma-separated hidden sizes")
    parser.add_argument("--device", type=str, default="cpu", help="device (cpu or cuda)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--attack-type", type=str, choices=["none", "fgsm", "pgd"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./benchmark_output")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    hidden_sizes = tuple(int(i) for i in args.hidden_sizes.split(",") if i.strip())

    run_centralized(
        dataset_path=args.dataset_path,
        num_rows=args.num_rows,
        hidden_sizes=hidden_sizes,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        attack_type=args.attack_type,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_iter=args.num_iter,
        random_start=args.random_start,
        output_dir=args.output_dir,
        seed=args.seed,
    )
