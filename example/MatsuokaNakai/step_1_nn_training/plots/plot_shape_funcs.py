import os
import pandas as pd
import matplotlib.pyplot as plt
pjoin = os.path.join

if __name__ == "__main__":
    func_dir = "example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func"
    # func_dir = "example/flower_shape/step_1_nn_training/results/cartesian/baseLO/PosEnc_1/shape_func"
    funcs = [0, 1, 2]
    for i in funcs:
        data = pd.read_csv(pjoin(func_dir, f'f_{i}_x_{i}.csv')).to_numpy()
        plt.plot(data[:, 0], data[:, 1])
        plt.show()