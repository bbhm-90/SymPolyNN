import os
from src.poly_nn.helper import read_data
from src.symbolic.regression_pysr import MyPySR
pjoin = os.path.join

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataAdd = "data/augmented_data_Bomarito_66k_noisy_4.csv"
    input_fields = ["sigma_h", "sigma_vm", "L", "v", "f"]
    outDir = pjoin(base_dir, "results")
    model_conf_add = pjoin(base_dir, "confg_pysr.json")
    output_dirpath = pjoin(base_dir, "results/")

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    
    xraw, yraw = read_data(dataAdd, input_fields=input_fields)
    
    model = MyPySR(model_conf_add=model_conf_add, output_dirpath=output_dirpath)
    model.fit_model(xraw, yraw)