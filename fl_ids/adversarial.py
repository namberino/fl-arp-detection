import numpy as np
import torch
import torch.nn as nn
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
        device: str = "cpu",
    ):
        self.model = model
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.random_start = random_start
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
    
    def fgsm_attack(self, X, y, training_mode=False):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Keep model in training mode if we're doing adversarial training
        if not training_mode:
            self.model.eval()
        
        X_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        gradient_sign = X_tensor.grad.data.sign()
        
        # Create adversarial examples (maximize loss by adding gradient)
        X_adv = X_tensor + self.epsilon * gradient_sign
        
        return X_adv.detach().cpu().numpy()
    
    def pgd_attack(self, X, y, training_mode=False):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Keep model in training mode if we're doing adversarial training
        if not training_mode:
            self.model.eval()
        
        X_adv = X_tensor.clone().detach()
        
        # Random initialization
        if self.random_start:
            X_adv = X_adv + torch.empty_like(X_adv).uniform_(-self.epsilon, self.epsilon)
            X_adv = torch.clamp(X_adv, X_tensor - self.epsilon, X_tensor + self.epsilon)
        
        # Iterative attack
        for _ in range(self.num_iter):
            X_adv.requires_grad = True
            
            # Forward pass
            outputs = self.model(X_adv)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Get gradient sign and update (maximize loss)
            gradient_sign = X_adv.grad.data.sign()
            X_adv = X_adv.detach() + self.alpha * gradient_sign
            
            # Project back to epsilon ball around original input
            X_adv = torch.clamp(X_adv, X_tensor - self.epsilon, X_tensor + self.epsilon)
        
        return X_adv.detach().cpu().numpy()
    
    def generate_adversarial_examples(self, X, y, training_mode=False):
        if self.attack_type == "fgsm":
            return self.fgsm_attack(X, y, training_mode)
        elif self.attack_type == "pgd":
            return self.pgd_attack(X, y, training_mode)
        else:
            return X
    
    def fit_with_adversarial(self, X, y, epochs=1, batch_size=32, learning_rate=0.001):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                if self.attack_type == "none":
                    # Standard training without adversarial examples
                    X_train = batch_X
                    y_train = batch_y
                else:
                    # Generate adversarial examples for this batch with current model weights
                    # Convert batch to numpy for attack generation
                    batch_X_np = batch_X.cpu().numpy()
                    batch_y_np = batch_y.cpu().numpy()
                    
                    # Generate adversarial examples (model stays in train mode)
                    X_adv_np = self.generate_adversarial_examples(
                        batch_X_np, batch_y_np, training_mode=True
                    )
                    
                    # Convert back to tensors
                    X_adv = torch.FloatTensor(X_adv_np).to(self.device)
                    
                    # Combine clean and adversarial examples
                    X_train = torch.cat([batch_X, X_adv], dim=0)
                    y_train = torch.cat([batch_y, batch_y], dim=0)
                
                # Forward pass
                outputs = self.model(X_train)
                loss = self.criterion(outputs, y_train)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            # print(f"  Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    def evaluate_robustness(self, X, y):
        self.model.eval()
        
        with torch.no_grad():
            # Clean accuracy
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            clean_acc = (predicted == y_tensor).sum().item() / len(y)
        
        # Adversarial accuracy
        if self.attack_type != "none":
            # Generate adversarial examples for evaluation
            X_adv = self.generate_adversarial_examples(X, y, training_mode=False)
            
            with torch.no_grad():
                X_adv_tensor = torch.FloatTensor(X_adv).to(self.device)
                outputs_adv = self.model(X_adv_tensor)
                _, predicted_adv = torch.max(outputs_adv, 1)
                adv_acc = (predicted_adv == y_tensor).sum().item() / len(y)
        else:
            adv_acc = clean_acc
        
        return {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "robustness_gap": clean_acc - adv_acc,
        }
