import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from datasets import mnist_digits
from nn_models import MLP, MLPConfig

X_train, y_train, metadata = mnist_digits.get_data_train_percevalquest()
X_val, y_val, metadata = mnist_digits.get_data_test_percevalquest()

lr = 0.01
num_epochs = 10
batch_size = 64

train_dataset = TensorDataset(torch.FloatTensor(X_train).reshape((-1,784)), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset( torch.FloatTensor(X_val).reshape((-1,784)), torch.LongTensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = MLP(input_size=784, output_size=10, config=MLPConfig(hidden_sizes=[64]))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

pbar = tqdm(range(num_epochs))

for epoch in pbar:
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    # Print epoch statistics
    pbar.set_description(f'Train Loss: {train_loss / len(train_loader):.4f} | '
                         f'Val Loss: {val_loss / len(val_loader):.4f} | '
                         f'Val Acc: {100 * correct / total:.2f}%')