import os
import numpy as np
import pandas as pd

from utils.plot_params import PlotParams
import matplotlib.pyplot as plt
pjoin = os.path.join

# def plot_penalty(data_dir:str, figDir:str):
#     penalty_dynamics = pd.read_csv(data_dir, index_col=0)
#     plot_params = PlotParams()
#     for c in penalty_dynamics.columns:
#         plt.plot(penalty_dynamics[c], label=f'w$_{{{c}}}$')
#     plt.xlabel("epochs", **plot_params.axis_font)
#     plt.ylabel("shape function weight", **plot_params.axis_font)
#     # plt.yscale("log")
#     plt.xscale("log")
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig(pjoin(figDir, 'shape_func_weight.png'))
#     plt.show()
#     plt.close()

def plot_penalty(data_dir:str, figDir:str):
    penalty_dynamics_add = pjoin(data_dir, "penalty_dynamics.csv")
    loss_pred_add = pjoin(data_dir, "loss_pred.csv")
    penalty_dynamics = pd.read_csv(penalty_dynamics_add, index_col=0)
    loss_pred = pd.read_csv(loss_pred_add, index_col=0)
    num_zeros = 0
    for c in penalty_dynamics.columns:
        if abs(penalty_dynamics[c].iloc[-1]) < 1e-4:
            num_zeros += 1
    alphas = np.linspace(0., 1., len(penalty_dynamics.columns) - num_zeros + 5)

    cntr = 0
    epoch = -1#int(1e4)
    fig, ax1 = plt.subplots()
    plot_params = PlotParams()
    for c in penalty_dynamics.columns:
        x = np.arange(1, len(penalty_dynamics[c].iloc[:epoch])+1)
        if abs(penalty_dynamics[c].iloc[-1]) < 1e-4:
            ax1.plot(x, penalty_dynamics[c].iloc[:epoch], label=f'w$_{{{c}}}$')
        else:
            ax1.plot(x, penalty_dynamics[c].iloc[:epoch], label=f'w$_{{{c}}}$', c='k', alpha=alphas[3+cntr])
            cntr += 1

    ax1.set_xscale("log")
    ax1.set_xlabel("epochs", **plot_params.axis_font)
    ax1.set_ylabel("shape function weight", **plot_params.axis_font)

    ax2 = ax1.twinx()
    ax2.plot(x, loss_pred.loss_pred[1:], 'r--')
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylabel('MSE level-set prediction', color='r')
    ax2.tick_params('y', colors='r')

    # plt.yscale("log")
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.legend()
    plt.tight_layout()
    # plt.savefig(pjoin(figDir, 'shape_func_weight.png'))
    plt.show()
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = "example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0"
    # data_dir = "example/MatsuokaNakai/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1"
    plot_penalty(data_dir, None)