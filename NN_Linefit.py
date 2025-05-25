import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

STAGE = 10
def dummy_cost(state: float) -> float:
    # A dummy cost function that returns a constant value
    return 0.0

class NN_Line_Fitting:
    def __init__(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> None:
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, x: float) -> torch.Tensor:
        x = torch.tensor([x], dtype=torch.float32)
        with torch.no_grad():
            return self.model(x).item()
    
    def __call__(self, x:float) -> torch.Any:
        x = torch.tensor([x], dtype=torch.float32)
        with torch.no_grad():
            return self.model(x).item()

def backward_pass(dict: Dict[Tuple[float, int], float]) -> List[NN_Line_Fitting]:
    dict = dict.copy()
    # Prepare data for each stage
    func_stage_x = [[] for _ in range(STAGE)]
    func_stage_y = [[] for _ in range(STAGE)]
    for key, value in dict.items():
        stage = key[1]
        func_stage_x[stage].append(key[0])
        func_stage_y[stage].append(value)
    
    # Train a neural network for each stage
    NN_models = []
    for stage in range(STAGE):
        x = torch.tensor(func_stage_x[stage], dtype=torch.float32).view(-1, 1)
        y = torch.tensor(func_stage_y[stage], dtype=torch.float32).view(-1, 1)
        model = NN_Line_Fitting()
        model.fit(x, y, epochs=100)
        NN_models.append(model)
    NN_models.append(NN_models[0])
    return NN_models
