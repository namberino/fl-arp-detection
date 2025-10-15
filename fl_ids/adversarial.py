import numpy as np
from typing import Literal


class AdversarialTrainer:
    def __init__(
        self,
        model,
        attack_type: Literal["fgsm", "pgd", "none"] = "none",
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_iter: int = 10,
        random_start: bool = True,
    ):
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.random_start = random_start
    
    def _compute_gradient(self, X, y):
        # Get predictions
        probs = self.model.predict_proba(X)
        
        # Get coefficients and handle binary vs multi-class
        coef = self.model.coef_
        n_classes = len(self.model.classes_)
        
        # Create one-hot encoded labels
        n_samples = len(y)
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # Gradient of cross-entropy loss
        error = probs - y_onehot  # Shape: (n_samples, n_classes)
        
        # Handle binary classification (coef shape is (1, n_features))
        if n_classes == 2 and coef.shape[0] == 1:
            # For binary classification, sklearn uses single coefficient vector
            # Convert to 2-class format
            coef_full = np.vstack([-coef[0], coef[0]])  # Shape: (2, n_features)
            gradient = error @ coef_full
        else:
            # Multi-class classification
            gradient = error @ coef  # Shape: (n_samples, n_features)
        
        return gradient
    
    def fgsm_attack(self, X, y):
        # Compute gradient
        gradient = self._compute_gradient(X, y)
        
        # Create adversarial examples
        X_adv = X + self.epsilon * np.sign(gradient)
        
        return X_adv
    
    def pgd_attack(self, X, y):
        X_adv = X.copy()
        
        # Random initialization
        if self.random_start:
            X_adv = X_adv + np.random.uniform(-self.epsilon, self.epsilon, X_adv.shape)
            X_adv = np.clip(X_adv, X - self.epsilon, X + self.epsilon)
        
        # Iterative attack
        for _ in range(self.num_iter):
            # Compute gradient
            gradient = self._compute_gradient(X_adv, y)
            
            # Update adversarial examples
            X_adv = X_adv + self.alpha * np.sign(gradient)
            
            # Project back to epsilon ball
            X_adv = np.clip(X_adv, X - self.epsilon, X + self.epsilon)
        
        return X_adv
    
    def generate_adversarial_examples(self, X, y):
        if self.attack_type == "fgsm":
            return self.fgsm_attack(X, y)
        elif self.attack_type == "pgd":
            return self.pgd_attack(X, y)
        else:
            return X
    
    def fit_with_adversarial(self, X, y):
        if self.attack_type == "none":
            # No adversarial training
            self.model.fit(X, y)
        else:
            # Generate adversarial examples
            X_adv = self.generate_adversarial_examples(X, y)
            
            # Combine clean and adversarial examples
            X_combined = np.vstack([X, X_adv])
            y_combined = np.hstack([y, y])
            
            # Train on combined dataset
            self.model.fit(X_combined, y_combined)
    
    def evaluate_robustness(self, X, y):
        # Clean accuracy
        clean_acc = self.model.score(X, y)
        
        # Adversarial accuracy
        if self.attack_type != "none":
            X_adv = self.generate_adversarial_examples(X, y)
            adv_acc = self.model.score(X_adv, y)
        else:
            adv_acc = clean_acc
        
        return {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "robustness_gap": clean_acc - adv_acc,
        }
