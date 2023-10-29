from LeNet import LeNet5
import matplotlib.pyplot as plt
from constants import *


def compare_max_trains(max_epochs=300):
    iterations=[]
    accuracy=[]
    net = LeNet5()
    test_result = net.train_and_test(max_epochs)
    for (i,acc) in test_result:
        iterations.append(i)
        accuracy.append(acc)
    plt.figure(figsize=(0.6*max_epochs/5-3, 5))
    plt.plot(iterations, accuracy, label='Accuracy', marker='o', color='#228B22')
    for i in range(len(iterations)):
        plt.annotate(f'{accuracy[i]:.2f}', (iterations[i], accuracy[i]), textcoords='offset points', xytext=(0, 10),
                     ha='center',fontsize=int(max_epochs/50)+6)
    plt.title('Accuracy on training epochs')
    plt.xlabel('Max_Epochs')
    plt.ylabel('Accuracy/%')
    plt.ylim(min(accuracy)-5, 100)
    plt.axhline(y=max(accuracy),color='#A0A0A0',linestyle='--',label=f'Max Accuracy:{max(accuracy):.2f}')
    plt.xticks(iterations)
    plt.legend()
    plt.savefig('max_train_result.png', dpi=300)


def compare_fc_layers(max_layer=4):
    fc_layer=[]
    max_accuracy=[]
    max_accuracy_value=[]
    for i in range(3,max_layer+1):
        fc_layer.append(i)
        net=LeNet5(fc_count=i)
        test_result=net.train(MAX_TRAIN+50*(i-2))
        max_acc=max(test_result,key=lambda x:x[1])
        max_accuracy.append(max_acc)
        max_accuracy_value.append(max_acc[1])
    plt.figure(figsize=(6,4))
    plt.plot(fc_layer,max_accuracy_value,label='Accuracy@ArgmaxEpoch',marker='o',color='#228B22')
    for i in range(len(fc_layer)):
        plt.annotate(f'{max_accuracy[i][1]:.2f}@{max_accuracy[i][0]}',(fc_layer[i],max_accuracy[i][1]),
                     textcoords='offset points',xytext=(0,10),ha='center',fontsize=8)
    plt.title('Accuracy on FC layers')
    plt.xlabel('FC layers')
    plt.ylabel('Accuracy/%')
    plt.ylim(min(max_accuracy_value)-5,100)
    plt.xlim(min(fc_layer)-1,max(fc_layer)+1)
    plt.xticks(fc_layer)
    plt.legend()
    plt.savefig('fc_compare_result.png',dpi=300)


def default_test():
    net = LeNet5(channel1=6, channel2=16, fc_count=3)
    net.train(30)
    total_acc,accuracies=net.test()
    _visualize_classes_accuracy(total_acc, accuracies, 'default_result.png')


def adjusted_test():
    net=LeNet5(channel1=6,channel2=24,fc_count=4)
    test_result=net.train_and_test(250,10)
    best_acc=max(test_result,key=lambda x:x[1])
    net=LeNet5(channel1=6,channel2=24,fc_count=4)
    net.train(best_acc[0])
    total_acc,accuracies=net.test()
    _visualize_classes_accuracy(total_acc, accuracies, 'adjusted_result.png')


def _visualize_classes_accuracy(total, accuracies, path):
    classes=[x[0] for x in accuracies]
    accuracies=[x[1] for x in accuracies]
    plt.figure(figsize=(6,4))
    plt.bar(classes,accuracies,color=DEFAULT_COLOR)
    for i in range(len(accuracies)):
        plt.annotate(f'{accuracies[i]:.2f}',xy=(classes[i],accuracies[i]),xytext=(0,10),textcoords='offset points',
                     ha='center',color=DEFAULT_COLOR,fontsize=8)
    plt.axhline(y=total,color=GRAY,linestyle='--',label='Total accuracy')
    plt.annotate(f'Total:{total:.2f}',xy=(0,total),xytext=(-30,0),textcoords='offset points',ha='right'
                 ,color=GRAY,fontsize=7)
    plt.title('Accuracy on classes')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy/%')
    plt.ylim(0,100)
    plt.legend()
    plt.savefig(path,dpi=300)


if __name__ == '__main__':
    default_test()
    adjusted_test()
    compare_max_trains(200)
    compare_fc_layers(6)


