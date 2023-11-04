import torch
import torchvision
import torchvision.transforms as transforms

MAX_TRAIN=55
BATCH_SIZE=64
CHANNEL_1=6
CHANNEL_2=16
FC_COUNT=3
TEST_GAP=5
GRAY='#A0A0A0'
DEFAULT_COLOR='#228B22'
MAX_ENTITIES=5

transform = transforms.Compose(
            [transforms.ToTensor(), # transforms the image data to tensor
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize RGB to [0,1]

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
TRAIN_LOADER = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=transform)
TEST_LOADER = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

