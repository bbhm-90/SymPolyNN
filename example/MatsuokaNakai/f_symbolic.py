import numpy as np
from src.symbolic.equation_assembler import SymbolicYeildSurfacePolynimialHO

config0 = {
    "form_type":"PolynomialHO",# str
    "x_num_dim":3, # int
    "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/xscaler.joblib", # pkl file path
    "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/yscaler.joblib", # pkl file path
    "func_weights": [
        1.3890066146850586, 2.176483631134033, 0.23617863655090332,
        -6.0072937735355936e-09, -0.2189200520515442, 9.166651437908513e-08,
        -6.200025826075262e-09, 3.1179769877098806e-09, -8.219394942443614e-09
    ], # list of float
    "ho_dim_pairs": [[0,1],[0,2],[1,2],[0,0],[1,1],[2,2]],
    "final_bias": 0.6875945925712585,
    "symb_funcs":
    [
        {
            "input_dims":[0], # list of int
            "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_0_x_0/tmp8kwixho0/xscaler.joblib", # pkl file path
            "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_0_x_0/tmp8kwixho0/yscaler.joblib", # pkl file path
            "equation":"((x0 / (-1.0003778 + (-0.3425479 * cos(sin(sin(exp(-0.3425479 * (sin((log(1.0482423) / x0) * 0.9900566) + -0.4054698)) / x0)))))) * 1.266024)", # str (complexity: 25)
        },
        {
            "input_dims":[1], # list of int
            "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_1_x_1/tmpsh7sudxh/xscaler.joblib", # pkl file path
            "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_1_x_1/tmpsh7sudxh/yscaler.joblib", # pkl file path
            "equation":"(x0 + (0.011891233 * ((sin(cos(x0 * (x0 + (cos(x0) * cos(x0))))) + (-0.31972098 / 0.9982944)) * 0.89531267)))", # str (complexity: 21)
        },
        {
            "input_dims":[2], # list of int
            "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_2_x_2/tmpbp82i60i/xscaler.joblib", # pkl file path
            "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_21/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_2_x_2/tmpbp82i60i/yscaler.joblib", # pkl file path
            "equation":"(((sin(-1.5393547 * ((x0 + 1.2935897) / 0.3214464)) + 0.13249452) / cos(cos(cos(sin(sin((1.5246235 * -1.5633299) * x0))) + ((1.5246235 * -1.5633299) * x0)) * -1.0388634)) + 0.031979296)", # str (complexity: 31)
        },
    ]
}

model_symb = SymbolicYeildSurfacePolynimialHO(config0)
def f_symbolic(p, rho, theta):
  
  RT = np.array([p, rho, theta]).reshape(1,3)
  f = model_symb.predict(RT).item()

  return f