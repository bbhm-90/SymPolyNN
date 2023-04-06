import os
import time
import joblib
import json
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.poly_nn.model import (
    MyActivation, 
    PolynomialHigherOrder,
    MyMLP
)
from src.poly_nn.helper_parser import (
    write_args_to_json,
    get_scaler,
    positive_float,
    nonnegative_float,
    positive_int,
    int_list,
    string_list
)
from src.poly_nn.helper import (
    read_data,
    check_args,
)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--addetive_class", type=str, required=True, help="which additive class to use")
    # data pre proc
    parser.add_argument("--outDir", type=str, required=True, help="full address of output directory")
    parser.add_argument("--dataAdd", type=str, required=True, help="full address of data with csv format")
    # parser.add_argument("--dataCord", type=str, required=True, help="data representation coordinate system for training")
    parser.add_argument("--dataCord", type=str, required=True, help="coordinate data")
    parser.add_argument("--xScalerType", type=str, required=True, help="scalar type for input")
    parser.add_argument("--yScalerType", type=str, required=True, help="scalar type for output")
    parser.add_argument("--final_bias", type=int, default=1, help="a learnable constant should be in the final form or not")
    parser.add_argument("--randomSeed", type=int, required=True, help="random seed, -1 means no control")

    ## neural network
    parser.add_argument("--layers_after_first", type=int_list, required=True, help="a list of number of nuerons per layer")
    parser.add_argument("--activations", type=lambda s: string_list(s, ','), required=True, help="a list of activation names per layer")
    parser.add_argument("--positional_encoding", action='store_true', default=False, help='need positional encoding')
    parser.add_argument("--sigma_pos_enc", type=nonnegative_float, default=0., help='standard deviation for positional encoding')

    ## optimization
    parser.add_argument("--lr", type=positive_float, required=True, help='adam initial learning rate')
    parser.add_argument("--epochs", type=positive_int, default=int(2e3), help='number of epochs')
    parser.add_argument("--stop_loss", type=positive_float, default=1e-5, help='stop epochs up to this loss')
    parser.add_argument("--batch_size", type=positive_int, required=True, help='batch size')
    parser.add_argument("--train_shuffle", action='store_true', default=False, help='shuffle batchs in training')
    parser.add_argument("--lrChangeFactor", type=positive_float, default=0.92, help='factor for lr change')
    parser.add_argument("--lrPatience", type=int, default=50, help='lr control patience')
    parser.add_argument("--lrStartEpoch", type=int, default=1000, help='when start lr control stepping')
    parser.add_argument("--updateOutputPeriod", type=int, default=1000, help='every how many epochs update outputs')
    parser.add_argument("--penalty_first_order", type=float, default=0., help='enforces sparsity in linear terms')
    parser.add_argument("--penalty_second_order", type=float, default=0., help='enforces sparsity in higher order terms')
    return parser

def save_train(model, loss_pred_record, loss_sprs_1st_record, loss_sprs_2nd_record, lr_op_record, weight_record, args):
    global TIME_INIT
    time_elapsed = time.time() - TIME_INIT
    with open(pjoin(args.outDir, "time_elapsed.txt"), "w") as f:
        f.write(f"{time_elapsed}")
    torch.save(model.state_dict(), pjoin(args.outDir, 'model.pth'))
    
    tmp = {'loss_pred': np.array(loss_pred_record)}
    if len(loss_sprs_1st_record) > 0:
        tmp['loss_sprs_1st'] = np.array(loss_sprs_1st_record)
    if len(loss_sprs_2nd_record) > 0:
        tmp['loss_sprs_2nd'] = np.array(loss_sprs_2nd_record)
    tmp['lr'] = np.array(lr_op_record)
    pd.DataFrame(tmp).to_csv(pjoin(args.outDir, 'loss_pred.csv'))
    # pd.DataFrame(tmp).to_csv(pjoin(args.outDir, 'loss_pred.csv'))
    # tmp = np.array(lr_op_record)
    # pd.DataFrame(tmp).to_csv(pjoin(args.outDir, 'lr.csv'))
    with torch.no_grad():
        tmp = {}
        if hasattr(model, 'weights_1st_order'):
            tmp['weights_1st_order'] = model.weights_1st_order.detach().numpy().flatten().tolist()
        if hasattr(model, 'weights_2nd_order'):
            tmp['weights_2nd_order'] = model.weights_2nd_order.detach().numpy().flatten().tolist()
        if hasattr(model, 'dim_pairs'):
            tmp['dim_pairs'] = model.dim_pairs.detach().numpy().tolist()
        tmp['final_bias'] = model.final_bias.item()
    if len(weight_record) >0:
        wnames = [str(i) for i in range(len(tmp['weights_1st_order']))]
        wnames += [str(ij[0])+str(ij[1]) for ij in tmp['dim_pairs']]
        pd.DataFrame(np.stack(weight_record, axis=0), columns=wnames).to_csv(pjoin(args.outDir, 'penalty_dynamics.csv'))
    with open(pjoin(args.outDir, 'func_weights.json'), 'w') as f:
        json.dump(tmp, f, indent=2)
    print("--- saved model")

def train(args):
    if args.randomSeed > 0:
        torch.manual_seed(args.randomSeed)
        np.random.seed(args.randomSeed)
    else:
        seed = np.random.randint(0, 10000, 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        args.randomSeed = seed

    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)
    if args.dataCord == "cartesian":
        input_fields = ['sigma1', 'sigma2', 'sigma3']
    elif args.dataCord == "cylindrical":
        input_fields = ['p', 'rho', 'theta']
    elif args.dataCord == "Bomarito":
        input_fields = ["sigma_h", "sigma_vm", "L", "v"]
    else:
        raise NotImplementedError(args.dataCord)

    xraw, yraw = read_data(args.dataAdd, input_fields=input_fields)

    indx = np.arange(xraw.shape[0])
    np.random.shuffle(indx)
    xraw = xraw[indx, :]
    yraw = yraw[indx, :]


    if args.addetive_class in {"baseLO", "PolynomialHO"}:
        addetiveClass = PolynomialHigherOrder
    elif args.addetive_class == "singleMLP":
        addetiveClass = MyMLP
    else:
        raise NotImplementedError(args.addetive_class)
    if args.addetive_class == "singleMLP":
        model = addetiveClass(layers=[xraw.shape[1]] + args.layers_after_first,
            activations=[MyActivation(act) for act in args.activations],
            positional_encoding=args.positional_encoding,
            params={'sigma_pos_enc':args.sigma_pos_enc, 'final_bias':args.final_bias}
        )
    else:
        model = addetiveClass(
            dim=xraw.shape[1], 
            layers_after_first=args.layers_after_first,
            activations=[MyActivation(act) for act in args.activations],
            positional_encoding=args.positional_encoding,
            params={'sigma_pos_enc':args.sigma_pos_enc, 'final_bias':args.final_bias}
        )

    yscaler = get_scaler(args.yScalerType)
    yscaler.fit(yraw)
    joblib.dump(yscaler, pjoin(args.outDir, "yscaler.joblib"))
    yscaled = yscaler.transform(yraw)

    xscaler = get_scaler(args.xScalerType)
    xscaler.fit(xraw)
    joblib.dump(xscaler, pjoin(args.outDir, "xscaler.joblib"))
    xscaled = xscaler.transform(xraw)

    dataTrn = np.concatenate([xscaled, yscaled], axis=1)

    dataTrn = torch.from_numpy(dataTrn).float()
    dataTrn = DataLoader(dataTrn, batch_size=args.batch_size, shuffle=args.train_shuffle)

    optE2E = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optE2E, factor=args.lrChangeFactor, patience=args.lrPatience
    )
    loss_pred_record = []
    loss_sprs_1st_record = []
    loss_sprs_2nd_record = []
    lr_op_record = []

    write_args_to_json(args, pjoin(args.outDir, "args.json"))

    best_pred = 1000.
    otherTrainParams = {"penalty_first_order":args.penalty_first_order, "penalty_second_order":args.penalty_second_order}
    weight_records = []
    global TIME_INIT; TIME_INIT = time.time()
    for epoch in range(args.epochs):
        loss_pred_epoch = 0.
        loss_sprs_1st_epoch = 0.
        loss_sprs_2nd_epoch = 0.
        tot_data = 0
        for xy in dataTrn:
            loss_sprs_1st = 0.
            loss_sprs_2nd = 0.
            if args.addetive_class == "singleMLP":
                loss_pred =\
                    model.train_regression(x=xy[:, :-1], y=xy[:, -1:],opt=optE2E)
            elif args.addetive_class == "baseLO":
                loss_pred, loss_sprs_1st =\
                    model.train_first_order_only(x=xy[:, :-1], y=xy[:, -1:],opt=optE2E, **otherTrainParams)
            else:
                loss_pred, loss_sprs_1st, loss_sprs_2nd =\
                    model.train_end_to_end(x=xy[:, :-1], y=xy[:, -1:],opt=optE2E, **otherTrainParams)
                
            loss_pred_epoch += (loss_pred*xy.shape[0])
            loss_sprs_1st_epoch += (loss_sprs_1st*xy.shape[0])
            loss_sprs_2nd_epoch += (loss_sprs_2nd*xy.shape[0])
            tot_data += xy.shape[0]
        with torch.no_grad():
            if hasattr(model, "weights_1st_order") and hasattr(model, "weights_2nd_order"):
                tmp = np.concatenate([model.weights_1st_order.detach().numpy().flatten(), 
                        model.weights_2nd_order.detach().numpy().flatten()])
                weight_records.append(tmp)
        loss_pred_epoch /= tot_data
        loss_sprs_1st_epoch /= tot_data
        loss_sprs_2nd_epoch /= tot_data
        loss_pred_record.append(loss_pred_epoch)
        loss_sprs_1st_record.append(loss_sprs_1st_epoch)
        loss_sprs_2nd_record.append(loss_sprs_2nd_epoch)
        cur_lr = optE2E.param_groups[0]['lr']
        lr_op_record.append(cur_lr)
        print(epoch, loss_pred_epoch, cur_lr)
        if best_pred > 1.1 * loss_pred_epoch and epoch%args.updateOutputPeriod == 0:
            save_train(model, loss_pred_record, loss_sprs_1st_record, loss_sprs_2nd_record, lr_op_record, weight_records, args)
        if epoch > args.lrStartEpoch:
            lr_sch.step(loss_pred_epoch)
        if loss_pred < args.stop_loss:
            break
    save_train(model, loss_pred_record, loss_sprs_1st_record, loss_sprs_2nd_record, lr_op_record, weight_records, args)

if __name__ == "__main__":
    pjoin = os.path.join
    parser = get_parser()
    args = parser.parse_args()
    check_args(args)
    train(args)