import numpy as np
from src.symbolic.equation_assembler import SymbolicYeildSurfacePolynimialHO

config0 = {
    "form_type":"PolynomialHO",# str
    "x_num_dim":4, # int
    "xscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/xscaler.joblib", # pkl file path
    "yscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/yscaler.joblib", # pkl file path
    "func_weights": [
    5.541553704802027e-08, 3.400454318125412e-08, -1.655053871729706e-08, 5.2695256158585835e-08,
    -10.277324676513672, -7.7271199226379395, 5.405412673950195, 
    -3.169267177581787, 1.0259310007095337, -1.1744688749313354,
    -2.4643006324768066, 1.4251837730407715, 3.173708438873291,
    3.2276546955108643,
    ], # list of float
    "final_bias": 0.0,
    "ho_dim_pairs": [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,0],[1,1],[2,2],[3,3]],
    "symb_funcs":
    [
        {
            "input_dims":[0], # list of int
            "xscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_0/tmp1o7yoqnr/xscaler.joblib", # pkl file path
            "yscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_0/tmp1o7yoqnr/yscaler.joblib", # pkl file path
            "equation":"(exp(0.049011476 * x0) * x0)", # str
        },
        {
            "input_dims":[1], # list of int
            "xscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_1/tmpu849fzrr/xscaler.joblib", # pkl file path
            "yscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_1/tmpu849fzrr/yscaler.joblib", # pkl file path
            "equation":"(x0 + (0.09338936 * cos(x0 * -1.8473725)))", # str
        },
        {
            "input_dims":[2], # list of int
            "xscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_2/tmpm63ru9d0/xscaler.joblib", # pkl file path
            "yscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_2/tmpm63ru9d0/yscaler.joblib", # pkl file path
            "equation":"(((-0.9930121 + (((x0 + x0) + cos(x0)) * 0.17261055)) * x0) + -0.34535986)", # str
        },
        {
            "input_dims":[3], # list of int
            "xscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_3/tmpmmps91zc/xscaler.joblib", # pkl file path
            "yscaler_path":"examples/Bomarito/PolyNAM_88k_noisy4/shape_func/symbolic_pysr/1st_3/tmpmmps91zc/yscaler.joblib", # pkl file path
            "equation":"(((cos(x0) / exp(1.2437834 + x0)) + x0) + ((-0.23015736 * x0) * x0))", # str
        },
    ]
}

model_symb = SymbolicYeildSurfacePolynimialHO(config0)
def f_symbolic(p, rho, theta, v):
  
  RT = np.array([p, rho, theta, v]).reshape(1,4)
  f = model_symb.predict(RT).item()

  return f