import os
import autograd.numpy as np
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

class MyPlasticity():
    def __init__(self, Fem):
        self.E  = Fem.E
        self.nu = Fem.nu
        self.model = SymbolicYeildSurfacePolynimialHO(config0)

    def f(self, sigma1, sigma2, sigma3, lamda):

        sigma1 = -sigma1
        sigma2 = -sigma2
        sigma3 = -sigma3

        sigma = np.array([sigma1, sigma2, sigma3])

        Rinv = np.array([[ np.sqrt(2)/2,             0, -np.sqrt(2)/2],
                         [-np.sqrt(6)/6,  np.sqrt(6)/3, -np.sqrt(6)/6],
                         [ np.sqrt(3)/3,  np.sqrt(3)/3,  np.sqrt(3)/3]])

        sigma_pp = np.dot(Rinv, sigma)

        sigma1_pp = sigma_pp[0]
        sigma2_pp = sigma_pp[1]
        sigma3_pp = sigma_pp[2]

        rho   = np.sqrt(sigma1_pp**2 + sigma2_pp**2)
        theta = np.arctan2(sigma2_pp, sigma1_pp)
        p     = (1/np.sqrt(3))*sigma3_pp

        if theta < 0:
            theta = theta + 2*np.pi

        RT = np.array([p, rho, theta]).reshape(1,3)
        f = self.model.predict(RT).item()

        return f

    def df(self, sigma1, sigma2, sigma3, lamda):

        dist = 1e-4
        f = self.f(sigma1, sigma2, sigma3, lamda)
        f_s1dist = self.f(sigma1+dist, sigma2, sigma3, lamda)
        f_s2dist = self.f(sigma1, sigma2+dist, sigma3, lamda)
        f_s3dist = self.f(sigma1, sigma2, sigma3+dist, lamda)

        dfdsig1  = (f_s1dist - f) / dist
        dfdsig2  = (f_s2dist - f) / dist
        dfdsig3  = (f_s3dist - f) / dist
        dfdlamda = 0.0

        norm = np.sqrt(dfdsig1**2 + dfdsig2**2 + dfdsig3**2)
        dfdsig1 = dfdsig1 / norm
        dfdsig2 = dfdsig2 / norm
        dfdsig3 = dfdsig3 / norm

        return dfdsig1, dfdsig2, dfdsig3, dfdlamda

    def df2(self, sigma1, sigma2, sigma3):

        dist = 1e-4
        dfdsig1, dfdsig2, dfdsig3, _ = self.df(sigma1, sigma2, sigma3, 0.0)

        dfdsig1_s1dist, dfdsig2_s1dist, dfdsig3_s1dist, _ = self.df(sigma1+dist, sigma2, sigma3, 0.0)
        dfdsig1_s2dist, dfdsig2_s2dist, dfdsig3_s2dist, _ = self.df(sigma1, sigma2+dist, sigma3, 0.0)
        dfdsig1_s3dist, dfdsig2_s3dist, dfdsig3_s3dist, _ = self.df(sigma1, sigma2, sigma3+dist, 0.0)

        d2fdsig1dsig1 = (dfdsig1_s1dist - dfdsig1) / dist
        d2fdsig2dsig2 = (dfdsig2_s2dist - dfdsig2) / dist
        d2fdsig3dsig3 = (dfdsig3_s3dist - dfdsig3) / dist

        d2fdsig1dsig2 = (dfdsig1_s2dist - dfdsig1) / dist
        d2fdsig2dsig3 = (dfdsig2_s3dist - dfdsig2) / dist
        d2fdsig3dsig1 = (dfdsig3_s1dist - dfdsig3) / dist

        return d2fdsig1dsig1, d2fdsig2dsig2, d2fdsig3dsig3, d2fdsig1dsig2, d2fdsig2dsig3, d2fdsig3dsig1