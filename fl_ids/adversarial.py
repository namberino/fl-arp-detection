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
    
    def fgsm_attack(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        X_tensor.requires_grad = True
        
        # Forward pass
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial examples
        gradient = X_tensor.grad.data
        X_adv = X_tensor + self.epsilon * gradient.sign()
        
        return X_adv.detach().cpu().numpy()
    
    def pgd_attack(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
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
            
            # Update adversarial examples
            gradient = X_adv.grad.data
            X_adv = X_adv.detach() + self.alpha * gradient.sign()
            
            # Project back to epsilon ball
            X_adv = torch.clamp(X_adv, X_tensor - self.epsilon, X_tensor + self.epsilon)
        
        return X_adv.detach().cpu().numpy()
    
    def generate_adversarial_examples(self, X, y):
        self.model.eval()
        
        if self.attack_type == "fgsm":
            return self.fgsm_attack(X, y)
        elif self.attack_type == "pgd":
            return self.pgd_attack(X, y)
        else:
            return X
    
    def fit_with_adversarial(self, X, y, epochs=1, batch_size=32, learning_rate=0.001):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if self.attack_type == "none":
            # No adversarial training - standard training
            X_train = X
            y_train = y
        else:
            # Generate adversarial examples
            X_adv = self.generate_adversarial_examples(X, y)
            
            # Combine clean and adversarial examples
            X_train = np.vstack([X, X_adv])
            y_train = np.hstack([y, y])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
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
            X_adv = self.generate_adversarial_examples(X, y)
            
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
