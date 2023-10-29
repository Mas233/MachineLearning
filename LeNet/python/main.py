from LeNet import LeNet5
import matplotlib.pyplot as plt
from constants import MAX_TRAIN

def compare_max_trains(max_epochs=150):
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
    for i in range(3,max_layer+1):
        net=LeNet5(fc_count=i)
        net.train(MAX_TRAIN+50*i)
        net.test()
    # visualize


if __name__ == '__main__':
    compare_max_trains()