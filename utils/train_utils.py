# Hàm huấn luyện
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from timeit import default_timer as timer
# Giúp hiển thị thanh tiến trình khi train
from tqdm.auto import tqdm
from configs.config_settings import config


def print_train_time(start, end, device: torch.device = None):
    total_time = end - start
    print(f"Training time: {total_time:.2f} seconds")
    return total_time 

def train_model(model, train_dataloader, test_dataloader, epochs, learning_rate, device):
    torch.manual_seed(42)
    train_time_start_on_cpu = timer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"We're using {device}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    accuracy_fn = Accuracy(task="multiclass", num_classes=len(train_dataloader.dataset.classes)).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\\n------------------------------")
        ###Training
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            model.train() # Kích hoạt training mode

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss (per patch)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            # 3. Optimizer zero_grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()
            
            # 5. Optimizer step
            optimizer.step()

            # In ra có bao nhiêu samples đã được train
            if batch % 400 == 0:
                print(f"Looked at {batch * config['batch_size']}/{len(train_dataloader.dataset)} samples")
            
        train_loss /= len(train_dataloader)

        ###Testing
        test_loss, test_acc = 0, 0
        model.eval() # Tắt BatchNorm & Dropout khi test
        with torch.inference_mode(): # Tắt gradient để tăng tốc độ
            for X, y in test_dataloader:
                # 1. Forward pass
                test_pred = model(X)

                # 2. Calculate loss (accumulate)
                test_loss += loss_fn(test_pred, y)

                # 3. Calculate accuracy 
                test_acc += accuracy_fn(test_pred, y)

            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)

        print(f"Train loss: {train_loss:.4f}, Test loss; {test_loss: .4f}, Test accuracy: {test_acc:.4f}")

    train_time_end_on_cpu = timer()
    print_train_time_model = print_train_time(start=train_time_start_on_cpu, end=train_time_end_on_cpu, device=str(next(model.parameters()).device))