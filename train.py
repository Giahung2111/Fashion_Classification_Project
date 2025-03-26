# Script huấn luyện
import torch
from torch.utils.data import DataLoader
from configs.config_settings import config
# from datasets.custom_dataset import CustomDataset
from models.model import FashionMNISTModelV0, FashionMNISTModelV1, FashionMNISTModelV2
from utils.train_utils import train_model
from utils.data_utils import get_data

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if __name__ == "__main__":
    torch.manual_seed(42) 
    train_data, test_data = get_data()
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    model = FashionMNISTModelV2(config['input_shape'], config['hidden_units'], len(train_data.classes))
    model = model.to(device)  # Đưa model lên GPU
    
    # Cần chỉnh sửa hàm train_model để xử lý device
    train_model(model, train_dataloader, test_dataloader, config['num_epochs'], config['learning_rate'], device)
    torch.save(model.state_dict(), "checkpoints/TinyVGG_Model.pth")
