#!/bin/sh
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
EXAMPLEPATH=$(dirname "$PPATH")
ROOTPATH=$(dirname "$EXAMPLEPATH")
DATAPATH="$ROOTPATH/data" 
AddetiveClassOptions=("PolynomialHO" "baseLO")
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
            python "$EXAMPLEPATH/trainner_nn.py"\
                --addetive_class "$addetive_class"\
                --outDir "$PPATH/results/$dataCord/$addetive_class/PosEnc_$cntrPosEnc/"\
                --dataAdd "$DATAPATH/augmented_data_MatsuokaNakai.csv"\
                --inputFields $inputFields\
                --xScalerType "minmax"\
                --yScalerType "standard"\
                --randomSeed "142"\
                --layers_after_first "20","10","10","1"\
                --activations "relu","relu","relu","tanh"\
                --epochs "5"\
                --lr "0.001"\
                --batch_size "1000"\
                --positional_encoding\
                --sigma_pos_enc "$sigma_pos_enc"\
                --train_shuffle\
                --penalty_first_order "0.001"\
                --penalty_second_order "0.01"\
                --updateOutputPeriod "1000"\
                --final_bias "0"
        done;
        ((cntrPosEnc++))
    done;
done;