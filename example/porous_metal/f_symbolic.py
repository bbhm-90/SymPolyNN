import numpy as np
from src.symbolic.equation_assembler import SymbolicYeildSurfacePolynimialHO

# With sparsity
# config0 = {
#     "form_type":"PolynomialHO",# str
#     "x_num_dim":4, # int
#     "xscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/xscaler.joblib", # pkl file path
#     "yscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/yscaler.joblib", # pkl file path
#     "func_weights": [
#     5.541553704802027e-08, 3.400454318125412e-08, -1.655053871729706e-08, 5.2695256158585835e-08,
#     -10.277324676513672, -7.7271199226379395, 5.405412673950195, 
#     -3.169267177581787, 1.0259310007095337, -1.1744688749313354,
#     -2.4643006324768066, 1.4251837730407715, 3.173708438873291,
#     3.2276546955108643
#     ], # list of float
#     "final_bias": 0.0,
#     "ho_dim_pairs": [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,0],[1,1],[2,2],[3,3]],
#     "symb_funcs":
#     [
#         {
#             "input_dims":[0], # list of int
#             "xscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_0_x_0/tmpgfa_9imo/xscaler.joblib", # pkl file path
#             "yscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_0_x_0/tmpgfa_9imo/yscaler.joblib", # pkl file path
#             "equation":"(x0 + sin((cos(((x0 * 0.9565414) + 0.06096601) + sin(0.38257736)) * (exp(x0) + x0)) * -0.06633134))", # str (complexity = 19)
#         },
#         {
#             "input_dims":[1], # list of int
#             "xscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_1_x_1/tmp2exb9002/xscaler.joblib", # pkl file path
#             "yscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_1_x_1/tmp2exb9002/yscaler.joblib", # pkl file path
#             "equation":"(x0 + (sin(-0.1545409 * sin(-1.0079175 + x0)) * (x0 + cos(sin(x0) + (0.029846042 * -2.4140942)))))", # str (complexity = 19)
#         },
#         {
#             "input_dims":[2], # list of int
#             "xscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_2_x_2/tmpdjgo27sn/xscaler.joblib", # pkl file path
#             "yscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_2_x_2/tmpdjgo27sn/yscaler.joblib", # pkl file path
#             "equation":"((((sin(sin(sin((cos(sin(cos(sin(exp(log(cos(sin(sin(0.15875988))))) + sin(sin(x0))) / 0.99727225) + x0)) + x0) / 1.7831751))) + 0.44848233) + -1.6344448) * x0) + -0.33669972)", # str (complexity = 33)
#         },
#         {
#             "input_dims":[3], # list of int
#             "xscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_3_x_3/tmp_12jry4m/xscaler.joblib", # pkl file path
#             "yscaler_path":"example/porous_metal/step_1_nn_training/results/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_3_x_3/tmp_12jry4m/yscaler.joblib", # pkl file path
#             "equation":"((sin(exp(((-0.40175155 * x0) / 0.87574494) + sin(0.35113916)) * sin(cos(x0))) + (-0.40175155 + (x0 + (0.056001645 * (((-0.43033838 * x0) + sin(-0.40175155)) / (-0.81918186 + 0.15043537)))))) + -0.24069726)", # str (complexity = 33)
#         },
#     ]
# }

# Without sparsity
config0 = {
    "form_type":"PolynomialHO",# str
    "x_num_dim":4, # int
    "xscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/xscaler.joblib", # pkl file path
    "yscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/yscaler.joblib", # pkl file path
    "func_weights": [
    2.8690872192382812, 8.192245483398438, -1.236699104309082, 1.2939486503601074,
    -6.3247761726379395, -4.657607555389404, 3.270129442214966,
    -2.2646052837371826, -0.31215739250183105, -0.010992146097123623,
    -0.48991909623146057, 0.760383665561676, 1.5245494842529297,
    1.7781329154968262
    ], # list of float
    "final_bias": 0.0,
    "ho_dim_pairs": [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,0],[1,1],[2,2],[3,3]],
    "symb_funcs":
    [
        {
            "input_dims":[0], # list of int
            "xscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_0_x_0/tmptxugi5rs/xscaler.joblib", # pkl file path
            "yscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_0_x_0/tmptxugi5rs/yscaler.joblib", # pkl file path
            "equation":"(x0 + (((sin(exp(x0 + (log(cos(0.664388)) * x0))) + -0.2977443) + -0.28641462) * -0.28641462))", # str (complexity = 17)
        },
        {
            "input_dims":[1], # list of int
            "xscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_1_x_1/tmpc1e8i3j4/xscaler.joblib", # pkl file path
            "yscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_1_x_1/tmpc1e8i3j4/yscaler.joblib", # pkl file path
            "equation":"(x0 + ((0.041742746 * (cos(x0) + cos(x0 + 0.9010846))) * ((x0 / sin(sin(0.8844295 / 1.0454644))) + 0.90298444)))", # str (complexity = 21)
        },
        {
            "input_dims":[2], # list of int
            "xscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_2_x_2/tmponc5oioi/xscaler.joblib", # pkl file path
            "yscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_2_x_2/tmponc5oioi/yscaler.joblib", # pkl file path
            "equation":"(((x0 * (-1.1978015 + (sin(x0 / exp(cos(cos(-0.7509346) * -1.0296835))) + (-0.0282291 * cos(x0))))) + sin(-0.49780723)) * cos(sin(cos((sin(sin(sin(-0.7215097))) * x0) + (-0.0282291 * cos(x0))))))", # str (complexity = 36)
        },
        {
            "input_dims":[3], # list of int
            "xscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_3_x_3/tmpjeuih8ao/xscaler.joblib", # pkl file path
            "yscaler_path":"example/porous_metal/step_1_nn_training/results_no_sparse/Bomarito/PolynomialHO/shape_func/symbolic_pysr/f_3_x_3/tmpjeuih8ao/yscaler.joblib", # pkl file path
            "equation":"((x0 + cos((x0 + cos(x0 / (-1.7659944 / ((cos(sin(sin(cos(x0 + ((x0 + cos(x0 / (1.5916637 / ((cos(sin(x0)) * x0) + cos(x0))))) + -0.56985813))))) * x0) + cos(x0))))) + -0.56985813)) + -0.56985813)", # str (complexity = 42)
        },
    ]
}


model_symb = SymbolicYeildSurfacePolynimialHO(config0)
def f_symbolic(p, rho, theta, v):
  
    sigma1pp = rho*np.cos(theta)
    sigma2pp = rho*np.sin(theta)
    sigma3pp = np.sqrt(3)*p

    sigma_pp = np.array([sigma1pp, sigma2pp, sigma3pp])

    R = np.array([[ np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3],
                  [            0,  np.sqrt(6)/3, np.sqrt(3)/3],
                  [-np.sqrt(2)/2, -np.sqrt(6)/6, np.sqrt(3)/3]])

    sigma = np.dot(R, sigma_pp)

    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]

    I1 = sigma1 + sigma2 + sigma3
    J2 = (1./6.)*((sigma1-sigma2)**2 + (sigma2-sigma3)**2 + (sigma3-sigma1)**2)
    
    sigma_h  = I1/3.
    sigma_vm = np.sqrt(3.*J2) 
    
    J3 = (sigma1 - sigma_h)*(sigma2 - sigma_h)*(sigma3 - sigma_h)

    L = (3.*np.sqrt(3)/2.)*(J3/(J2**1.5))

    RT = np.array([sigma_h, sigma_vm, L, v]).reshape(1,4)
    f = model_symb.predict(RT).item()

    return f