import numpy as np
from src.symbolic.equation_assembler import SymbolicYeildSurfacePolynimialHO

config0 = {
    "form_type":"PolynomialHO",# str
    "x_num_dim":3, # int
    "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/xscaler.joblib", # pkl file path
    "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/yscaler.joblib", # pkl file path
    "func_weights": [
        8.483229637145996, 6.242868900299072, 3.4870903491973877,
        -3.2097339630126953, -2.875276565551758, 4.639223098754883,
        -5.364171504974365, -0.6363519430160522, 2.2510015964508057
    ], # list of float
    "ho_dim_pairs": [[0,1],[0,2],[1,2],[0,0],[1,1],[2,2]],
    "final_bias": 2.1608002185821533,
    "symb_funcs":
    [
        {
            "input_dims":[0], # list of int
            "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_0_x_0/tmpmdzmutqu/xscaler.joblib", # pkl file path
            "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_0_x_0/tmpmdzmutqu/yscaler.joblib", # pkl file path
            "equation":"(((-1.0268155 + (0.18257223 * (sin(sin(cos(x0))) + x0))) * x0) + -0.18156151)", # str (complexity: 14)
        },
        {
            "input_dims":[1], # list of int
            "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_1_x_1/tmpykbgzpl5/xscaler.joblib", # pkl file path
            "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_1_x_1/tmpykbgzpl5/yscaler.joblib", # pkl file path
            "equation":"(x0 + (sin((x0 + (x0 + 1.08937)) + cos(x0)) * 0.034534745))", # str (complexity: 13)
        },
        {
            "input_dims":[2], # list of int
            "xscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_2_x_2/tmpbevwy471/xscaler.joblib", # pkl file path
            "yscaler_path":"example/MatsuokaNakai/step_1_nn_training/results_1/cylindrical/PolynomialHO/PosEnc_0/shape_func/symbolic_pysr/f_2_x_2/tmpbevwy471/yscaler.joblib", # pkl file path
            "equation":"(sin(sin(sin(sin(sin((x0 / -0.20826437) + (0.11208224 * cos(0.83082455))) + (0.15613697 / 0.9782789))))) / sin(sin(sin(cos(sin((x0 / -0.20826437) + (0.15613697 * sin(log(exp(0.5182119))))) + sin(-0.20826437))))))", # str (complexity: 36)
        },
    ]
}

model_symb = SymbolicYeildSurfacePolynimialHO(config0)
def f_symbolic(p, rho, theta, lamda):
  
  RT = np.array([p, rho, theta]).reshape(1,3)
  f = model_symb.predict(RT).item()

  return f