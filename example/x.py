import json
add = "example/flower_shape/step_1_nn_training/results/cylindrical/baseLO/PosEnc_0/args.json"
with open(add, 'r') as f:
    data = json.load(f)
print(type(data['layers_after_first'][0]))