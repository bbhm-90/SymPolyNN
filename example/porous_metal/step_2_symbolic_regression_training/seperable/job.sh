#!/bin/bash
SEP_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
SYMTRN_PATH=$(dirname "$SEP_PATH")
FLOW_PATH=$(dirname "$SYMTRN_PATH")
EX_PATH=$(dirname "$FLOW_PATH")
ROOT_PATH=$(dirname "$EX_PATH")

AddetiveClassOptions=("PolynomialHO" "baseLO")
SigmaPosEnc=("2.")
# DataCord=("cylindrical" "cartesian")
DataCord=("Bomarito")

for addetive_class in ${AddetiveClassOptions[*]};do
    for dataCord in ${DataCord[*]};do            
        shFuncDir="$FLOW_PATH/step_1_nn_training/results/$dataCord/$addetive_class/shape_func/"
        python "$EX_PATH/trainner_symb.py"\
            --func_dir $shFuncDir\
            --conf_pysr $SEP_PATH/confg_pysr.json\
            --num_trials 1
    done;
done;