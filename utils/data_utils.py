# Hàm tiền xử lý dữ liệu
import pandas as pd
from torchvision import datasets
from torchvision.transforms import ToTensor

def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

def get_data():
    train_data = datasets.FashionMNIST(root='data/processed', train = True, download=True, transform = ToTensor(), target_transform=None)
    test_data = datasets.FashionMNIST(root='data/processed', train = False, download=True, transform = ToTensor())
    return train_data, test_data


