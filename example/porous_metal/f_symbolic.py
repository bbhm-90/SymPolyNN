import numpy as np
from src.symbolic.equation_assembler import SymbolicYeildSurfacePolynimialHO

config0 = {
    "form_type":"PolynomialHO",# str
    "x_num_dim":4, # int
    "xscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/xscaler.joblib", # pkl file path
    "yscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/yscaler.joblib", # pkl file path
    "func_weights": [
        0.43378180265426636, 5.274606704711914, 3.827580213546753,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], # list of float
    "ho_dim_pairs": [[0,1],[0,2],[1,2],[0,0],[1,1],[2,2]],
    "final_bias": 0.0,
    "symb_funcs":
    [
        {
            "input_dims":[0], # list of int
            "xscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_0_x_0/tmpsr7h3hr4/xscaler.joblib", # pkl file path
            "yscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_0_x_0/tmpsr7h3hr4/yscaler.joblib", # pkl file path
            "equation":"0.", # str (complexity: 1)
        },
        {
            "input_dims":[1], # list of int
            "xscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_1_x_1/tmpp1f48rdr/xscaler.joblib", # pkl file path
            "yscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_1_x_1/tmpp1f48rdr/yscaler.joblib", # pkl file path
            "equation":"(x0 + ((sin(sin((sin(sin(x0)) + (x0 / 1.2935598)) + 0.29268932)) * cos(x0 / sin(-0.8599956))) * -0.0076511777))", # str (complexity: 21)
        },
        {
            "input_dims":[2], # list of int
            "xscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_2_x_2/tmpc0fmkpui/xscaler.joblib", # pkl file path
            "yscaler_path":"example/flower_shape/step_1_nn_training/results_2/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_2_x_2/tmpc0fmkpui/yscaler.joblib", # pkl file path
            "equation":"(((sin(-4.829514 * x0) + cos((((sin(-4.829514 * x0) + cos(-0.8067878 * sin(sin(-4.829514 * x0)))) + -0.8067878) * 1.3604934) * cos(cos(log(exp(exp(1.2876843))))))) + -0.8067878) * 1.3604934)", # str (complexity: 34)
        },
    ]
}

model_symb = SymbolicYeildSurfacePolynimialHO(config0)
def f_symbolic(p, rho, theta, v):
  
  RT = np.array([p, rho, theta, v]).reshape(1,4)
  f = model_symb.predict(RT).item()

  return f