import torch

def evaluate_model(model: torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, accuracy_fn, device: torch.device = device):
    model.eval() # Tắt BatchNorm & Dropout khi test
    loss, acc = 0, 0
    with torch.inference_mode(): # Tắt gradient để tăng tốc độ
        for X, y in dataloader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred, y)
        
        loss /= len(dataloader)
        acc /= len(dataloader)
        return  {"model_name": model.__class__.__name__, "model_loss": loss, "model_accuracy": acc}