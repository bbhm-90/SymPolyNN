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
    data_name = "augmented_data_flower"
    dataCord = "cylindrical"
    form_type = "baseLO"
    
    raw_data = "data/augmented_data_flower_26k.csv"

    loss0 = pjoin(res_dir, f"cylindrical/{form_type}/PosEnc_1/loss_pred.csv")
    loss1 = pjoin(res_dir, f"cartesian/{form_type}/PosEnc_1/loss_pred.csv")

    yraw_pred0 = pjoin(res_dir, f"cylindrical/{form_type}/PosEnc_1/yraw_pred.csv")
    yraw_pred1 = pjoin(res_dir, f"cartesian/{form_type}/PosEnc_1/yraw_pred.csv")

    raw_data = pd.read_csv(raw_data).to_numpy()[:, -1]
    loss0 = pd.read_csv(loss0).to_numpy()[:, 1]
    yraw_pred0 = pd.read_csv(yraw_pred0).to_numpy()[:, -1]
    loss1 = pd.read_csv(loss1).to_numpy()[:, 1]
    yraw_pred1 = pd.read_csv(yraw_pred1).to_numpy()[:, -1]

    plot_params = PlotParams()
    plt.plot(loss1, label='cartesian')
    plt.plot(loss0, label='cylindrical')
    plt.xlabel("epochs", **plot_params.axis_font)
    plt.ylabel("MSE loss prediction", **plot_params.axis_font)
    plt.xticks(**plot_params.axis_ticks)
    plt.yticks(**plot_params.axis_ticks)
    plt.yscale("log")
    plt.tight_layout()
    plt.legend(**plot_params.legend_font)
    # plt.savefig(pjoin(out_fig, 'pr1_comp_loss_cyldCords_vs_cartCord_spec.png'))
    plt.show()
    plt.close()

    tmp = np.concatenate((yraw_pred0, yraw_pred0, raw_data))
    tmp = get_equal_yx_line(
        min(tmp), 
        max(tmp)
    )
    plot_params = PlotParams()
    plt.plot(tmp[:, 0],tmp[:, 1], 'k--', alpha=0.6)

    plt.scatter(yraw_pred1, raw_data, alpha=0.3, s=10, label='cartesian')
    plt.scatter(yraw_pred0, raw_data, alpha=0.3, s=10, label='cylindrical')
    plt.xlabel("predicted level-set", **plot_params.axis_font)
    plt.ylabel("data level-set", **plot_params.axis_font)
    plt.xticks(**plot_params.axis_ticks)
    plt.yticks(**plot_params.axis_ticks)
    plt.tight_layout()
    plt.legend(**plot_params.legend_font)
    # plt.savefig(pjoin(out_fig, 'pr1_comp_levelSet_cyldCords_cartCords_spec.png'))
    plt.show()
    plt.close()