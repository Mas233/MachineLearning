from LeNet import LeNet5
import matplotlib.pyplot as plt

def compare_max_trains():
    iterations = [15 * i + 10 for i in range(6)]
    accuracy = []
    for i in iterations:
        net = LeNet5()
        net.train(i)
        accuracy.append(net.test())
    plt.plot(iterations, accuracy, label='Accuracy', marker='o', color='#228B22')
    for i in range(len(iterations)):
        plt.annotate(f'{accuracy[i]:.2f}', (iterations[i], accuracy[i]), textcoords='offset points', xytext=(0, 10),
                     ha='center')
    plt.title('Accuracy on training epochs')
    plt.xlabel('Max_Epochs')
    plt.ylabel('Accuracy/%')
    plt.ylim(0, 100)
    plt.xticks(iterations)
    plt.legend()
    plt.savefig('result.png', dpi=300)


if __name__=='__main__':
    pass