#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
EXAMPLEPATH=$(dirname "$PPATH")
EXAMPLEPATH=$(dirname "$EXAMPLEPATH")
ROOTPATH=$(dirname "$EXAMPLEPATH")
DATAPATH="$ROOTPATH/data" 
# AddetiveClassOptions=("baseLO" "PolynomialHO")
# SigmaPosEnc=("2." "0.")
# DataCord=("cylindrical")
AddetiveClassOptions=("PolynomialHO")
SigmaPosEnc=("2.")
DataCord=("cylindrical")

for addetive_class in ${AddetiveClassOptions[*]};do
    cntrPosEnc=0
    for sigma_pos_enc in ${SigmaPosEnc[*]};do
        for dataCord in ${DataCord[*]};do
            if [ $dataCord == "cylindrical" ]; then
                inputFields=("p" "rho" "theta")
            elif [ $dataCord == "cartesian" ]; then
                inputFields=("sigma1" "sigma2" "sigma3")
            fi
            echo "$dataCord $addetive_class $sigma_pos_enc $cntrPosEnc"
            outDir="$PPATH/results_4/$dataCord/$addetive_class/PosEnc_$cntrPosEnc/"
            python "$EXAMPLEPATH/trainner_nn.py"\
                --addetive_class "$addetive_class"\
                --outDir $outDir\
                --dataAdd "$DATAPATH/augmented_data_MatsuokaNakai_13k.csv"\
                --dataCord $dataCord\
                --xScalerType "minmax"\
                --yScalerType "standard"\
                --randomSeed "142"\
                --layers_after_first "40","40","40","1"\
                --activations "relu","relu","relu","iden"\
                --epochs "10000"\
                --lr "0.001"\
                --batch_size "5000"\
                --positional_encoding\
                --sigma_pos_enc "$sigma_pos_enc"\
                --train_shuffle\
                --penalty_first_order "0.0"\
                --penalty_second_order "0.0"\
                --updateOutputPeriod "300"\
                --final_bias "0"
            python "$EXAMPLEPATH/post_process_trained_nn.py" $outDir
        done;
        ((cntrPosEnc++))
    done;
done;