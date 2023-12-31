import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
import numpy as np
from constants import *


class LeNet5(nn.Module):
    def __init__(self,channel1=CHANNEL_1,channel2=CHANNEL_2,fc_count=FC_COUNT):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, channel1, 5,1,0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channel1, channel2, 5, 1, 0)
        self.fc1=nn.Linear(channel2 * 25, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc=[]
        for i in range(fc_count-3):
            self.fc.append(nn.Linear(84,84))
        self.fc3=nn.Linear(84, 10)
        self.channel2=channel2
        print(f'Init completed with channel1={channel1},channel2={channel2},fc_count={len(self.fc)+3}')

    def forward(self, x):
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = x.view(-1, self.channel2 * 25)
        x = Func.relu(self.fc1(x))
        x = Func.relu(self.fc2(x))
        for i in range(len(self.fc)):
            x = Func.relu(self.fc[i](x))
        x = self.fc3(x)
        return x

    def train(self,  max_epochs=MAX_TRAIN):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, data in enumerate(TRAIN_LOADER, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, EntropyLoss: {running_loss / len(TRAIN_LOADER):.6f}")
        print("Training Finished")

    def train_and_test(self,max_epochs=MAX_TRAIN,gap=TEST_GAP,train_loader=TRAIN_LOADER):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        accuracies=[]
        low_avg_case=0
        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.7f}")
            if (epoch + 1) % gap == 0:
                acc,_=self.test()
                accuracies.append((epoch+1,acc))

                # avoid over-fitting
                if (len(accuracies)>7):
                    last_mean = np.mean([x[1] for x in accuracies[-7:-5]])
                    this_mean = np.mean([x[1] for x in accuracies[-3:-1]])
                    print(f'last_mean:{last_mean:.2f}% this_mean:{this_mean:.2f}%')
                    if(last_mean>=this_mean):
                        low_avg_case+=1
                        print(f'{low_avg_case} low cases')
                        if low_avg_case>1:
                            break

        print("Training Finished")
        return accuracies

    def test(self,test_loader=TEST_LOADER):
        accuracies=[]
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()

                # Update class-wise counts.
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(10):
            accuracies.append((CLASSES[i],class_correct[i]*100.0/class_total[i]))
            print(f'  {accuracies[i][0]} accuracy: {accuracies[i][1]:.2f}%')
        total_accuracy=sum(class_correct)*100.0/sum(class_total)
        print(f'Total accuracy:{total_accuracy:.2f}%')
        return total_accuracy,accuracies
