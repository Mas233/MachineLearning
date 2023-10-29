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
        self.fc=[nn.Linear(channel2 * 25, 120), nn.Linear(120, 84)]
        for i in range(fc_count-3):
            self.fc.append(nn.Linear(84,84))
        self.fc.append(nn.Linear(84, 10))
        self.dataset=DATASET()

    def forward(self, x):
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        x = x.view(-1, CHANNEL_2 * 25)
        for fc in self.fc:
            x=Func.relu(fc(x))
        return x


    def train(self,  max_epochs=MAX_TRAIN):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataset.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)  # Call the forward method explicitly
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.dataset.train_loader):.3f}")
        print("Finished Training")


    def train_and_test(self,max_epochs=MAX_TRAIN):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        accuracies=[]
        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.dataset.train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.forward(inputs)  # Call the forward method explicitly
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(self.dataset.train_loader):.3f}")
            if (epoch + 1) % 5 == 0:
                accuracies.append((epoch+1,self.test()))
        print("Finished Training")
        return accuracies


    def test(self):
        # Initialize variables to store class-wise correct predictions and total samples.
        class_correct = list(0. for _ in range(10))
        class_total = list(0. for _ in range(10))

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

        # Calculate and print the overall accuracy.
        total_accuracy = 100 * sum(class_correct) / sum(class_total)
        print(f'Overall accuracy: {total_accuracy:.2f}%')
        return total_accuracy
