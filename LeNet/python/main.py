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
    plt.figure(figsize=(0.6*max_epochs/5-3, 6))
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
    for i in range(3,max_layer+1):
        fc_layer.append(i)
        net=LeNet5(fc_count=i)
        test_result=net.train_and_test(MAX_TRAIN+100*(i-3))
        max_acc=max(test_result,key=lambda x:x[1])
        max_accuracy.append(max_acc)
    _visualize_args_accuracy(fc_layer,max_accuracy,'FC layers','fc_compare_result.png')


def compare_channels(max_channel=32,step=4):
    channel=[]
    max_accuracy=[]
    for i in range(16,max_channel+step-1,step):
        channel.append(i)
        net=LeNet5(channel1=CHANNEL_1+2*((i-16)//step),channel2=i)
        test_result=net.train_and_test(MAX_TRAIN+50*(i-16))
        max_acc=max(test_result,key=lambda x:x[1])
        max_accuracy.append(max_acc)
    _visualize_args_accuracy(channel,max_accuracy,'Feature Channels','channel2_compare_result.png')


def default_test():
    net = LeNet5(channel1=6, channel2=16, fc_count=3)
    net.train(30)
    total_acc,accuracies=net.test()
    _visualize_classes_accuracy(total_acc, accuracies, 'default_result.png')


def adjusted_test():
    net=LeNet5(channel1=6,channel2=32,fc_count=5)
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
        plt.annotate(f'{accuracies[i]:.2f}',xy=(classes[i],accuracies[i]),xytext=(0,10),
                     textcoords='offset points',ha='center',color=DEFAULT_COLOR,fontsize=8)
    plt.axhline(y=total,color=GRAY,linestyle='--',label='Total accuracy')
    plt.annotate(f'Total:{total:.2f}',xy=(0,total),xytext=(-30,0),textcoords='offset points',ha='right'
                 ,color=GRAY,fontsize=7)
    plt.title('Accuracy on classes')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy/%')
    plt.ylim(0,100)

    _custom_adjust_x_axis(classes)

    plt.legend()
    plt.savefig(path,dpi=300)


def _visualize_args_accuracy(arg_arr, accuracies, arg_name, path):
    plt.figure(figsize=(6,4))
    plt.bar(arg_arr, [x[1] for x in accuracies], color=DEFAULT_COLOR, label='Accuracy@ArgmaxEpoch')
    for i in range(len(accuracies)):
        plt.annotate(f'{accuracies[i][1]:.2f}@{accuracies[i][0]}', xy=(arg_arr[i], accuracies[i][1]),
                     xytext=(0, 10), textcoords='offset points',ha='center', color=DEFAULT_COLOR, fontsize=8)
    plt.title(f'Accuracy on {arg_name}')
    plt.ylabel('Accuracy/%')
    plt.ylim(0,100)
    plt.xlabel(arg_name)

    _custom_adjust_x_axis(arg_arr)

    plt.legend()
    plt.savefig(path,dpi=300)


# adjust ticks on x-axis
def _custom_adjust_x_axis(arr):
    x_ticks=arr
    x_ticks.insert(0,min(arr)-1)
    x_ticks.append(max(arr) + 1)
    plt.xticks(x_ticks)
    plt.gca().get_xaxis().get_major_ticks()[0].label1.set_visible(False)
    plt.gca().get_xaxis().get_major_ticks()[-1].label1.set_visible(False)


if __name__ == '__main__':
    compare_channels(40)
    compare_fc_layers(7)

