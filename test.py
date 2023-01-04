import torch
from torch.utils.data import DataLoader

from net import net
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

data_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='/data', train=True, transform=data_transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)


test_dataset = datasets.MNIST(root='/data', train=False, transform=data_transform, download=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = net().to(device)


model.load_state_dict(torch.load('C:/Users/崔岩cy/Desktop/论文/2d论文/LeNet/model_path/last_model.pth'))

classes = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
]

# tensor -> 图片
show = ToPILImage()
for i in range(20):
    x, y = test_dataset[i][0], test_dataset[i][1]
    show(x).show()

    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'preddicted:"{predicted}", actual:"{actual}"')