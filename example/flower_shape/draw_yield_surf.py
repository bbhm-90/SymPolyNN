# Import necessary packages and functions
import os
import sys
import torch
import torch.nn as nn
import joblib
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import autograd.numpy as np
from autograd import elementwise_grad as egrad

# Data points
N_points = 100
theta = np.linspace(0, 2*np.pi, N_points+1)[0:N_points]


# Newton-Raphson details
maxiter = 10
tol = 1e-11


# Benchmark yield function
from example.flower_shape.benchmark import *
get_dfdrho = egrad(f_benchmark, 1)


# NAM yield function
model_NAM = torch.jit.load("example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/model.ptjit")
f_INPUT_scaler  = joblib.load("example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/xscaler.joblib")
f_OUTPUT_scaler = joblib.load("example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/yscaler.joblib")

def f_NAM(p, rho, theta, lamda):

  RT = np.array([p, rho, theta]).reshape(1,3)
  RT = f_INPUT_scaler.transform(RT)
  RT = torch.tensor(RT, dtype=torch.float)
  
  f = model_NAM(RT)
  f = f[0]
  f_numpy = f.detach().numpy()[0,0]
  f_numpy = f_OUTPUT_scaler.inverse_transform(f_numpy.reshape(-1,1))

  return f_numpy[0,0]


# NAM-symbolic yield function
from src.symbolic.equation_assembler import SymbolicYeildSurfacePolynimialHO

config0 = {
    "form_type":"PolynomialHO",# str
    "x_num_dim":3, # int
    "xscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/xscaler.joblib", # pkl file path
    "yscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/yscaler.joblib", # pkl file path
    "func_weights": [
        0.43378180265426636, 5.274606704711914, 3.827580213546753,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ], # list of float
    "ho_dim_pairs": [[0,1],[0,2],[1,2],[0,0],[1,1],[2,2]],
    "symb_funcs":
    [
        {
            "input_dims":[0], # list of int
            "xscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_0_x_0/tmpsr7h3hr4/xscaler.joblib", # pkl file path
            "yscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_0_x_0/tmpsr7h3hr4/yscaler.joblib", # pkl file path
            "equation":"0.", # str (complexity: 1)
        },
        {
            "input_dims":[1], # list of int
            "xscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_1_x_1/tmpp1f48rdr/xscaler.joblib", # pkl file path
            "yscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_1_x_1/tmpp1f48rdr/yscaler.joblib", # pkl file path
            "equation":"(x0 + ((sin(sin((sin(sin(x0)) + (x0 / 1.2935598)) + 0.29268932)) * cos(x0 / sin(-0.8599956))) * -0.0076511777))", # str (complexity: 21)
        },
        {
            "input_dims":[2], # list of int
            "xscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_2_x_2/tmpc0fmkpui/xscaler.joblib", # pkl file path
            "yscaler_path":"example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_1/shape_func/symbolic_pysr/f_2_x_2/tmpc0fmkpui/yscaler.joblib", # pkl file path
            "equation":"(((sin(-4.829514 * x0) + cos((((sin(-4.829514 * x0) + cos(-0.8067878 * sin(sin(-4.829514 * x0)))) + -0.8067878) * 1.3604934) * cos(cos(log(exp(exp(1.2876843))))))) + -0.8067878) * 1.3604934)", # str (complexity: 34)
        },
    ]
}

model_symb = SymbolicYeildSurfacePolynimialHO(config0)
def f_symb(p, rho, theta, lamda):
  
  RT = np.array([p, rho, theta]).reshape(1,3)
  f = model_symb.predict(RT).item()

  return f


# Return mapping for benchmark yield function
rho = np.zeros_like(theta)

for i in range(np.shape(theta)[0]):

  x = 300.0

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_benchmark(0.0, x, theta[i], 0.0)
    jac = get_dfdrho(0.0, x, theta[i], 0.0)

    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err)

    if err < tol:
      rho[i] = x
      break


# Return mapping for NAM yield function
rho_NAM = np.zeros_like(theta)

for i in range(np.shape(theta)[0]):

  x = 300.0

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_NAM(0.0, x, theta[i], 0.0)
    jac = 1 # just used constant
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err, ", x =",x)

    if err < tol or ii == maxiter-1:
      rho_NAM[i] = x
      break


# Return mapping for NAM-symbolic yield function
rho_symb = np.zeros_like(theta)

for i in range(np.shape(theta)[0]):

  x = 300.0

  print(">> Point", i, "------------------------------------")

  for ii in range(maxiter):
    res = f_NAM(0.0, x, theta[i], 0.0)
    jac = 1 # just used constant
    
    dx = -res / jac
    x = x + dx

    err = np.linalg.norm(dx)

    print(" Newton iter.",ii, ": err =", err, ", x =",x)

    if err < tol or ii == maxiter-1:
      rho_symb[i] = x
      break


# Plot results
fig = plt.figure(0,figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, rho, 'k-')
ax.plot(theta, rho_NAM, 'r-')
ax.plot(theta, rho_symb, 'b-')
plt.show()