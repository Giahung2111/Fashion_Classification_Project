from utils import evaluate_utils
from models.model import FashionMNISTModelV0
from configs.config_settings import config
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from utils.data_utils import get_data
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Lấy dữ liệu
    train_data, test_data = get_data()
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # Các thiết lập khác
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = Accuracy(task="multiclass", num_classes=len(train_dataloader.dataset.classes)).to(device)

    # Load model
    loaded_model_0 = FashionMNISTModelV0(input_shape=config['input_shape'], hidden_units=config['hidden_units'], output_shape=len(train_dataloader.dataset.classes))
    loaded_model_0.load_state_dict(torch.load('checkpoints/model.pth'))

    #In ra kết quả đánh giá mô hình
    results = evaluate_utils.evaluate_model(loaded_model_0, test_dataloader, loss_fn, accuracy_fn=accuracy_fn)
    print(results)