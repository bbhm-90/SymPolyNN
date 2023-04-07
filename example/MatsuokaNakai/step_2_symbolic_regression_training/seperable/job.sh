#!/bin/bash
SEP_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
SYMTRN_PATH=$(dirname "$SEP_PATH")
FLOW_PATH=$(dirname "$SYMTRN_PATH")
EX_PATH=$(dirname "$FLOW_PATH")
ROOT_PATH=$(dirname "$EX_PATH")

AddetiveClassOptions=("PolynomialHO")
SigmaPosEnc=("2.")
# DataCord=("cylindrical" "cartesian")
DataCord=("cylindrical")

for addetive_class in ${AddetiveClassOptions[*]};do
    cntrPosEnc=0
    for sigma_pos_enc in ${SigmaPosEnc[*]};do
        for dataCord in ${DataCord[*]};do            
            # # echo "$dataCord $addetive_class $sigma_pos_enc $cntrPosEnc"
            shFuncDir="$FLOW_PATH/step_1_nn_training/results_1/$dataCord/$addetive_class/PosEnc_$cntrPosEnc/shape_func/"
            python "$EX_PATH/trainner_symb.py"\
                --func_dir $shFuncDir\
                --conf_pysr $SEP_PATH/confg_pysr.json\
                --num_trials 1
        done;
        ((cntrPosEnc++))
    done;
done;