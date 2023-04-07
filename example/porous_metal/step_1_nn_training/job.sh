#!/bin/bash
PPATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
EXAMPLEPATH=$(dirname "$PPATH")
EXAMPLEPATH=$(dirname "$EXAMPLEPATH")
ROOTPATH=$(dirname "$EXAMPLEPATH")
DATAPATH="$ROOTPATH/data" 
AddetiveClassOptions=("PolynomialHO" "baseLO")
DataCord=("Bomarito")

for addetive_class in ${AddetiveClassOptions[*]};do
    for dataCord in ${DataCord[*]};do
        echo "$dataCord $addetive_class" 
        outDir="$PPATH/results/$dataCord/$addetive_class/"
        python "$EXAMPLEPATH/trainner_nn.py"\
            --addetive_class "$addetive_class"\
            --outDir $outDir\
            --dataAdd "$DATAPATH/augmented_data_Bomarito_88k_noisy_4.csv"\
            --dataCord $dataCord\
            --xScalerType "minmax"\
            --yScalerType "standard"\
            --randomSeed "142"\
            --layers_after_first "10","10","1"\
            --activations "elu","elu","tanh"\
            --epochs "10"\
            --lr "0.001"\
            --batch_size "5000"\
            --positional_encoding\
            --train_shuffle\
            --penalty_first_order "0.01"\
            --penalty_second_order "0.001"\
            --updateOutputPeriod "100"\
            --final_bias "0"
        python "$EXAMPLEPATH/post_process_trained_nn.py" $outDir
    done;
done;