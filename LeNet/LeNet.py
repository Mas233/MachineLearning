import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from constants import MAX_TRAIN,DATASET

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        self.dataset=DATASET()

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


    def train(self,criterion=nn.CrossEntropyLoss):
        optimizer=optim.SGD(self.parameters(),lr=0.001,momentum=0.9)
        for epoch in range(MAX_TRAIN):
            running_loss=0.0
            for i,data in enumerate(self.dataset.train_loader,0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1} Loss:{running_loss / len(self.dataset.trainloader)}")



if __name__ == '__main__':


    net=LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # training
    for epoch in range(MAX_TRAIN):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        print(f"Epoch {epoch+1} Loss:{running_loss/len(trainloader)}")
    print("Finished Training")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total}%")

