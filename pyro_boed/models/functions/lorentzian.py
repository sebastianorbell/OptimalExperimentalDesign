import torch

def lorentzian(x, a, b):
    mean = a / ((x - b) ** 2 + a ** 2)
    return mean
