# Script đánh giá
import torch
from datasets.custom_dataset import CustomDataset
from models.model import NeuralNet

model = NeuralNet(23, 128, 1)
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()
