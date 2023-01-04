import torch
from torch import nn
from net import net
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


data_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='/data', train=True, transform=data_transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)


test_dataset = datasets.MNIST(root='/data', train=False, transform=data_transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = net().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        # forward
        x, y = x.to(device), y.to(device)
        output = model(x)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        # backward
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()

        current += cur_acc.item()
        n = n + 1
    print('train_loss' + str(loss/n))
    print('train_acc' + str(current/n))

def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # forward
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)

            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()

            current += cur_acc.item()
            n = n + 1
        print('test_loss' + str(loss / n))
        print('test_acc' + str(current / n))

    return current / n

epoch = 5

for t in range(epoch):
    print(f'epoch{t+1}\n--------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    print(a)

torch.save(model.state_dict(), 'model_path/last_model.pth')

print('down')














