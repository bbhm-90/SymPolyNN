#!/bin/bash
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
EXAMPLEPATH=$(dirname "$PPATH")
EXAMPLEPATH=$(dirname "$EXAMPLEPATH")
ROOTPATH=$(dirname "$EXAMPLEPATH")
DATAPATH="$ROOTPATH/data" 
AddetiveClassOptions=("baseLO")
SigmaPosEnc=("2." "0.")
# DataCord=("cylindrical" "cartesian")
DataCord=("cylindrical")

for addetive_class in ${AddetiveClassOptions[*]};do
    cntrPosEnc=0
    for sigma_pos_enc in ${SigmaPosEnc[*]};do
        for dataCord in ${DataCord[*]};do
            echo "$dataCord $addetive_class $sigma_pos_enc $cntrPosEnc" 
            outDir="$PPATH/results_5k/$dataCord/$addetive_class/PosEnc_$cntrPosEnc/"
            python "$EXAMPLEPATH/trainner_nn.py"\
                --addetive_class "$addetive_class"\
                --outDir $outDir\
                --dataAdd "$DATAPATH/augmented_data_flower_5k.csv"\
                --dataCord $dataCord\
                --xScalerType "minmax"\
                --yScalerType "standard"\
                --randomSeed "142"\
                --layers_after_first "20","20","1"\
                --activations "elu","elu","tanh"\
                --epochs "10000"\
                --lr "0.003"\
                --batch_size "10000"\
                --positional_encoding\
                --sigma_pos_enc "$sigma_pos_enc"\
                --train_shuffle\
                --penalty_first_order "0.0"\
                --penalty_second_order "0.0"\
                --updateOutputPeriod "1000"\
                --final_bias "0"
            python "$EXAMPLEPATH/post_process_trained_nn.py" $outDir
        done;
        ((cntrPosEnc++))
    done;
done;