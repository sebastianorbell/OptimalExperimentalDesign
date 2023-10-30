import functools
import operator

def get_variable_names(model_dict):
    variable_names = functools.reduce(
        operator.iconcat,
        [[sub_key for sub_key in item.keys()] for item in model_dict.values()],
        [])
    return variable_names

def deep_copy_dict(distribution_dict):
    copy_dict = {key: {sub_key: sub_value.detach().clone() for sub_key, sub_value in value.items()} for key, value in distribution_dict.items()}
    return copy_dict