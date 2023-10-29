import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
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
        self.dataset=Dataset()
        print(f'Init completed with channel1={channel1},channel2={channel2},fc_count={len(self.fc)+3}')

    def forward(self, x):
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = x.view(-1, self.channel2 * 25)
        x = Func.relu(self.fc1(x))
        x = Func.relu(self.fc2(x))
        for i in len(self.fc):
            x = Func.relu(self.fc[i](x))
        x = self.fc3(x)
        return x

    def train(self,  max_epochs=MAX_TRAIN):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataset.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, EntropyLoss: {running_loss / len(self.dataset.train_loader):.6f}")
        print("Training Finished")

    def train_and_test(self,max_epochs=MAX_TRAIN,gap=TEST_GAP):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        accuracies=[]
        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataset.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.dataset.train_loader):.6f}")
            if (epoch + 1) % gap == 0:
                acc,_=self.test()
                accuracies.append((epoch+1,acc))
        print("Training Finished")
        return accuracies

    def test(self):
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))
        accuracies=[]
        with torch.no_grad():
            for data in self.dataset.test_loader:
                inputs, labels = data
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()

                # Update class-wise counts.
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # Calculate and print the accuracy for each class.
        for i in range(10):
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of {self.dataset.classes[i]}: {accuracy:.2f}%')
            accuracies.append((self.dataset.classes[i],accuracy))

        # Calculate and print the overall accuracy.
        total_accuracy = 100 * sum(class_correct) / sum(class_total)
        print(f'Overall accuracy: {total_accuracy:.2f}%')
        return total_accuracy,accuracies
