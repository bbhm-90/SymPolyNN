import sys
import os
import pandas as pd
from src.poly_nn.helper_post_proces import PostProcessor

pjoin = os.path.join

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ppcl = PostProcessor(sys.argv[1])
    yscaled_pred = ppcl.get_prediction(ppcl.xscaled)
    yraw_pred = ppcl.get_inverse_transform_y(yscaled_pred)
    shape_functions_1st_order = ppcl.get_1st_shape_func(ppcl.xscaled)
    shape_functions_2nd_order =  ppcl.get_2nd_shape_func(ppcl.xscaled)
    ppcl.extract_shape_functions()
    pd.DataFrame(yraw_pred).to_csv(pjoin(ppcl.outDir, "yraw_pred.csv"))
