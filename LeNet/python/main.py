from LeNet import LeNet5
import matplotlib.pyplot as plt
from constants import *
import numpy as np


def compare_max_trains(max_epochs=200):
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


def compare_fc_layers(max_layer=4,max_entities=MAX_ENTITIES):
    fc_layer=[i for i in range(3,max_layer+1)]
    max_accuracy=[]
    for i in fc_layer:
        nets=[LeNet5(fc_count=i,channel1=CHANNEL_1+(i-3)*2,channel2=CHANNEL_2+16*(i-3)) for _ in range(max_entities)]
        test_results=[nets[j].train_and_test(MAX_TRAIN+100*(i-3)) for j in range(max_entities)]
        max_accs=[max(test_results[j],key=lambda x:x[1]) for j in range(max_entities)]
        max_acc=(sum([x[0] for x in max_accs])/max_entities,sum([x[1] for x in max_accs])/max_entities)
        max_accuracy.append(max_acc)
    _visualize_args_accuracy(fc_layer,max_accuracy,'FC layers','fc_compare_result.png')


def compare_channels(max_channel=32,step=4,max_entities=MAX_ENTITIES):
    channel=[i for i in range(16,max_channel+step-1,step)]
    max_accuracy=[]
    for i in channel:
        nets=[LeNet5(channel1=CHANNEL_1+2*((i-16)//step),channel2=i) for _ in range(max_entities)]
        test_results=[nets[j].train_and_test(MAX_TRAIN+50*(i-16)) for j in range(max_entities)]
        max_accuracy.append(_get_avg_max_accuracy(test_results,max_entities))
    _visualize_args_accuracy(channel,max_accuracy,'Feature Channels','channel2_compare_result.png')


def default_test():
    _batch_test(path='default_result.png')


def adjusted_test(channel1=18,channel2=40,fc_count=4,max_entities=MAX_ENTITIES):

    _batch_test(channel1=channel1,
                channel2=channel2,
                fc_count=fc_count,
                max_train=45,
                max_entities=max_entities,
                path='adjusted_result.png')


def _batch_test(channel1=CHANNEL_1,channel2=CHANNEL_2,fc_count=FC_COUNT,max_train=MAX_TRAIN,max_entities=MAX_ENTITIES,path='null.png',visualize=True):
    nets = [LeNet5(channel1=channel1, channel2=channel2, fc_count=fc_count) for _ in range(max_entities)]
    total_accuracies = []
    class_accuracies = []
    for net in nets:
        net.train(max_train)
        total_accuracy, class_accuracy = net.test()
        total_accuracies.append(total_accuracy)
        class_accuracies.append(class_accuracy)
    total_acc = np.mean(total_accuracies)
    accuracies = [(class_accuracies[0][i][0], np.mean([x[i][1] for x in class_accuracies])) for i in range(10)]
    if visualize:
        _visualize_classes_accuracy(total_acc, accuracies, path)
    return total_acc,accuracies


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
    if len(arr)>0 and isinstance(arr[0],(int,float,complex)):
        x_ticks.insert(0, min(arr) - 1)
        x_ticks.append(max(arr) + 1)
    else:
        x_ticks.insert(0, '')
        x_ticks.append('')
    plt.xticks(x_ticks)
    plt.gca().get_xaxis().get_major_ticks()[0].label1.set_visible(False)
    plt.gca().get_xaxis().get_major_ticks()[-1].label1.set_visible(False)


def _get_avg_max_accuracy(results,max_entities):
    max_accs = [max(results[j], key=lambda x: x[1]) for j in range(max_entities)]
    return sum([x[0] for x in max_accs]) / max_entities, sum([x[1] for x in max_accs]) / max_entities


if __name__ == '__main__':
    adjusted_test()
    compare_fc_layers(6)