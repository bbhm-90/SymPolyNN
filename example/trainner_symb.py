import os
import pandas as pd
from argparse import ArgumentParser
from src.symbolic.regression_pysr import MyPySR

pjoin = os.path.join

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--num_trials", type=int, required=True, help="number of training")
    parser.add_argument("--func_dir", type=str, required=True, help="functions directory")
    parser.add_argument("--conf_pysr", type=str, required=True, help="config file address")
    return parser

def run_pysr(func_add, conf_pysr):
    assert func_add.endswith(".csv"), func_add
    func_name = func_add.split("/")[-1].split(".")[0]
    par_dir = os.path.dirname(os.path.realpath(func_add))
    par_dir = pjoin(par_dir, f"symbolic_pysr/{func_name}")
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    data = pd.read_csv(func_add).to_numpy()[1:]
    clsym = MyPySR(conf_pysr, par_dir)
    elapsed_time = clsym.fit_model(data[:, :-1], data[:, -1:])
    return elapsed_time

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    func_dir = args.func_dir
    for fname in os.listdir(func_dir):
        if not fname.endswith(".csv"):
            continue
        data_path = pjoin(func_dir, fname)
        for _ in range(args.num_trials):
            run_pysr(data_path, args.conf_pysr)