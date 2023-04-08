import os
import numpy as np
import pandas as pd
from src.symbolic.regression_pysr import MyPySR
from utils.plot_params import PlotParams
from utils.symbolic import save_latex_table
import matplotlib.pyplot as plt
pjoin = os.path.join

def read_and_plot(data_sym_res_dir, loss_th=None):
    for sr_dir in os.listdir(data_sym_res_dir):
        if sr_dir == ".DS_Store":
            continue
        cur_res_dir = pjoin(data_sym_res_dir, sr_dir)
        config_path = pjoin(cur_res_dir, "config.json")
        for tmp in os.listdir(cur_res_dir):
            if tmp.endswith(".pkl"):
                pkl_path = pjoin(cur_res_dir, tmp)
                break
        xscalerPath = pjoin(cur_res_dir, "xscaler.joblib")
        yscalerPath = pjoin(cur_res_dir, "yscaler.joblib")
        model = MyPySR(config_path)
        model = model.load_from_file(pkl_path)
        model.load_scaler(xscalerPath, yscalerPath)
        x = pd.read_csv(data_path).to_numpy()
        x, y = x[:, 0:1], x[:, -1:]
        xfiner = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
        # for i in range(min(eq_index, len(model.get_all_equations()))):
        all_equs = model.get_all_equations()
        plot_params = PlotParams()
        plot_params.legend_font['prop']['size'] = 12
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="neural network", c='k', marker='.')
        totNumEq = len(model.equations_)
        # defaultOutIndices = set([1, 2, totNumEq//2, totNumEq-2, totNumEq-1])
        # defaultOutIndices = [1, 15, 35]
        # defaultOutIndices = [1, 4, 13]
        defaultOutIndices = [1, 10, 21]
        expr_list = []
        eq_rec = dict()
        eq_rec['x'] = xfiner.flatten()
        for i in range(totNumEq):
            # if loss_th is not None:
            #     if all_equs.iloc[i]['loss'] > loss_th:
            #         continue
            # else:
            #     if i not in defaultOutIndices:
            #         continue
            y_pred = model.get_predict(x, index=i)
            y_pred_finer = model.get_predict(xfiner, index=i)
            
            loss = ((y_pred - y)**2).mean()
            # print(y_pred)
            # print(model.get_sympy_expression(index=eq_index))
            # plt.plot(x, y_pred, label=model.get_sympy_expression(index=i))
            label = all_equs.iloc[i]['complexity']
            ax.plot(xfiner, y_pred_finer, label=f"symbolic eq. (complx. {label})")
            # ax.plot(x, y_pred, label=f"symbolic equation")
            expr = model.get_sympy_expression(index=i, num_decimals=2)
            expr_list.append((f'{label}', loss, expr))                
            # if loss_th is not None:
            #     break
            eq_rec[f'eq_complex_{label}'] = y_pred_finer.flatten()
        pd.DataFrame(eq_rec).to_csv(pjoin(cur_res_dir, "eqs_vs_complex.csv"))
        plt.xlabel("input", **plot_params.axis_font)
        plt.ylabel("shape function", **plot_params.axis_font)
        plt.xticks(**plot_params.axis_ticks)
        plt.yticks(**plot_params.axis_ticks)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), **plot_params.legend_font)
        ax.legend(**plot_params.legend_font)
        # plt.legend()
        plt.tight_layout()
        img_add = pjoin(cur_res_dir, "pr1_pred-vs-gt.png")
        plt.savefig(img_add)
        plt.close()
        print(img_add)
        # plt.show()
        save_latex_table(expr_list, pjoin(cur_res_dir, "pred-eqs.tex"))
        print(pjoin(cur_res_dir, "pr1_pred-eqs.tex"))

if __name__ == "__main__":
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # base_dir = pjoin(base_dir, "results/cylindrical/PolynomialHO/PosEnc_1/shape_func/")
    base_dir = "example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func"
    # func_names = ["f_0_x_0", "f_1_x_1", "f_2_x_2"]
    func_names = ["f_0_x_0", "f_1_x_1", "f_2_x_2"]
    loss_ths = [None] * len(func_names)
    # loss_ths = [0.413, 4.895e-5, 0.00031]
    # loss_ths = [0.413]
    # complex_levels = [5, 11]

    
    # data_path = "examples/MatsuokaNakai/results/cylindrical/PolynomialHO/PosEnc_0/shape_func/1st_0.csv"
    # data_sym_res_dir = "examples/MatsuokaNakai/results/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/1st_0"
    for loss_th, func_name in zip(loss_ths, func_names):
        data_path = pjoin(base_dir, f"{func_name}.csv")
        data_sym_res_dir = pjoin(base_dir, f"symbolic_pysr/{func_name}")
        read_and_plot(data_sym_res_dir, None, )