# Script dự đoán dữ liệu mới
import torch
from models.model import NeuralNet

model = NeuralNet(23, 128, 1)
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()

input = torch.tensor([[0.1, 0.2, ..., 0.5]])
output = model(input)
print(output)
