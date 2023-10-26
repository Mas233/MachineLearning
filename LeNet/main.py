from LeNet import LeNet5
import matplotlib.pyplot as plt

if __name__=='__main__':
    iterations = [25*i+50 for i in range(6)]
    accuracy = []
    for i in iterations:
        net=LeNet5()
        net.train(i)
        accuracy.append(net.test())
    plt.plot(iterations,accuracy,label='Accuracy',marker='o',color='#228B22')
    plt.title('Accuracy on training epochs')
    plt.xlabel('Max_Epochs')
    plt.ylabel('Accuracy/%')
    plt.ylim(0,100)
    plt.xticks(iterations)
    plt.legend()
    plt.savefig('result.png',dpi=300)