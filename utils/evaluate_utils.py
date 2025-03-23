import torch

def evaluate_model(model: torch.nn.Module, test_dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, accuracy_fn):
    model.eval() # Tắt BatchNorm & Dropout khi test
    loss, acc = 0, 0
    with torch.inference_mode(): # Tắt gradient để tăng tốc độ
        for X, y in test_dataloader:
            test_pred = model(X)
            loss += loss_fn(test_pred, y)
            acc += accuracy_fn(test_pred, y)
        
        loss /= len(test_dataloader)
        acc /= len(test_dataloader)
        return  {"model_name": model.__class__.__name__, "model_loss": loss, "model_accuracy": acc}