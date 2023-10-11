import torch.nn as nn
import copy


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        sequential=None,
        input_size=0,
        hidden_layers=0,
        output_size=0,
    ) -> None:
        super(NeuralNetwork, self).__init__()
        if not sequential:
            self.sequential = nn.Sequential(
                nn.Linear(
                    in_features=input_size,
                    out_features=hidden_layers,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=hidden_layers,
                    out_features=output_size,
                ),
            )
        else:
            self.sequential = sequential
        self.target: nn.Sequential = copy.deepcopy(self.sequential)
        self.target = self.target.requires_grad_(False)

    def forward(self, input, model: str = "online"):
        if model == "online":
            return self.sequential(input)
        elif model == "target":
            return self.target(input)
