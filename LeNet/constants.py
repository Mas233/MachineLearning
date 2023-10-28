import torch
import torchvision
import torchvision.transforms as transforms

MAX_TRAIN=55


class DATASET:

    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=64,
                                                        shuffle=True, num_workers=2)

        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                     download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64,
                                                       shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
