import os
import numpy as np
import torch
from src.poly_nn.model import (
    MyMLP,
    MyActivation
)
import matplotlib.pyplot as plt
torch.manual_seed(42)

if __name__ == "__main__":
    save_dir = "example/ablation/random_data"
    x = torch.linspace(-1., 1., 100).reshape(-1, 1)
    y = (torch.rand(100).reshape(-1, 1) - 0.5) * 4
    # plt.scatter(x, y)
    # plt.show()

    model = MyMLP(
        layers=[1, 80, 1],
        activations=[MyActivation(i) for i in ['relu', 'iden']],
        positional_encoding=False,
    )

    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_pred = []
    for epoch in range(int(10e3)):
        epoch_loss = model.train_regression(x, y, optim)
        print(epoch, epoch_loss)
        loss_pred.append(epoch_loss)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    plt.scatter(x, y, c=default_colors[0], label="data")
    plt.plot(x, model(x).detach(), c=default_colors[1], label="model")
    plt.xlabel("input")
    plt.ylabel("output")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "results/vanilla.png"))
    # plt.show()
