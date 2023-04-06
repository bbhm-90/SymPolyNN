from typing import Union
from argparse import (
    ArgumentParser,
    ArgumentTypeError
)
import json
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

def positive_int(value):
    try:
        float_value = int(value)
        if float_value <= 0:
            raise ArgumentTypeError("{} is an invalid positive int value".format(value))
        return float_value
    except ValueError:
        raise ArgumentTypeError("{} is not a valid int value".format(value))

def positive_float(value):
    try:
        float_value = float(value)
        if float_value <= 0:
            raise ArgumentTypeError("{} is an invalid positive float value".format(value))
        return float_value
    except ValueError:
        raise ArgumentTypeError("{} is not a valid float value".format(value))

def nonnegative_float(value):
    try:
        float_value = float(value)
        if float_value < 0:
            raise ArgumentTypeError("{} is an invalid positive float value".format(value))
        return float_value
    except ValueError:
        raise ArgumentTypeError("{} is not a valid float value".format(value))

def int_list(value):
    try:
        values = list(map(int, value.split(',')))
        return values
    except ValueError:
        raise ArgumentTypeError("{} is not a valid list of integers".format(value))

def string_list(value, separator=','):
    try:
        values = value.split(separator)
        return values
    except ValueError:
        raise ArgumentTypeError("{} is not a valid list of strings".format(value))

def write_args_to_json(args:ArgumentParser, file_add:str, indent=4):
    with open(file_add, 'w') as f:
        json.dump(args.__dict__, f, indent=indent)

# def load_args_from_json(file_add:str) -> ArgumentParser:
#     parser = ArgumentParser()
#     args = parser.parse_args()
#     with open(file_add, 'r') as f:
#         args.__dict__ = json.load(f)
#     return args

def get_scaler(scalerType:str) -> Union[StandardScaler, MinMaxScaler]:
    if scalerType == "standard":
        return StandardScaler()
    elif scalerType == "minmax":
        return MinMaxScaler()
    else:
        raise NotADirectoryError(scalerType)