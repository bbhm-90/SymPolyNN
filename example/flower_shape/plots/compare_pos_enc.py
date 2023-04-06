import os
import pandas as pd
import numpy as np

from utils.plot_params import PlotParams
from src.poly_nn.helper import get_equal_yx_line
import matplotlib.pyplot as plt
pjoin = os.path.join

if __name__ == "__main__":
    res_dir = "example/flower_shape/step_1_nn_training/results"
    out_fig = res_dir
    code_dir = '/'.join(res_dir.split("/")[:-3])
    data_name = "augmented_data_flower"
    dataCord = "cylindrical"
    form_type = "baseLO"
    
    raw_data = "data/augmented_data_flower.csv"
    loss0 = pjoin(res_dir, f"{dataCord}/{form_type}/PosEnc_0/loss_pred.csv")
    yraw_pred0 = pjoin(res_dir, f"{dataCord}/{form_type}/PosEnc_0/yraw_pred.csv")
    loss1 = pjoin(res_dir, f"{dataCord}/{form_type}/PosEnc_1/loss_pred.csv")
    yraw_pred1 = pjoin(res_dir, f"{dataCord}/{form_type}/PosEnc_1/yraw_pred.csv")
    # examples/vonMisesFlower/results/cylindrical/PolynomialHO/PosEnc_0/loss_pred.csv

    raw_data = pd.read_csv(raw_data).to_numpy()[:, -1]
    loss0 = pd.read_csv(loss0).to_numpy()[:, 1]
    yraw_pred0 = pd.read_csv(yraw_pred0).to_numpy()[:, -1]
    loss1 = pd.read_csv(loss1).to_numpy()[:, 1]
    yraw_pred1 = pd.read_csv(yraw_pred1).to_numpy()[:, -1]

    plot_params = PlotParams()
    plt.plot(loss0, label='without spectral layer')
    plt.plot(loss1, label='with spectral layer')
    plt.xlabel("epochs", **plot_params.axis_font)
    plt.ylabel("MSE loss prediction", **plot_params.axis_font)
    plt.xticks(**plot_params.axis_ticks)
    plt.yticks(**plot_params.axis_ticks)
    plt.yscale("log")
    plt.tight_layout()
    plt.legend(**plot_params.legend_font)
    # plt.savefig(pjoin(out_fig, 'pr1_comp_loss_spec_vs_nspec_cyldCords.png'))
    plt.show()
    plt.close()

    tmp = np.concatenate((yraw_pred0, yraw_pred0, raw_data))
    tmp = get_equal_yx_line(
        min(tmp), 
        max(tmp)
    )
    plot_params = PlotParams()
    plt.plot(tmp[:, 0],tmp[:, 1], 'k--', alpha=0.5)

    plt.scatter(yraw_pred0, raw_data, alpha=0.3, s=10, label='without spectral layer')
    plt.scatter(yraw_pred1, raw_data, alpha=0.3, s=10, label='with spectral layer')
    plt.xlabel("predicted level-set", **plot_params.axis_font)
    plt.ylabel("data level-set", **plot_params.axis_font)
    plt.xticks(**plot_params.axis_ticks)
    plt.yticks(**plot_params.axis_ticks)
    plt.tight_layout()
    plt.legend(**plot_params.legend_font)
    # plt.savefig(pjoin(out_fig, 'pr1_comp_levelSet_spec_vs_nspec_cyldCords.png'))
    # plt.show()
    plt.close()