import re
from matplotlib import pyplot as plt


if __name__=='__main__':
    with open(r'C:\Users\zhww\Desktop\log.txt','r') as f:
        str = f.read()

    train_acc = re.findall(r'Train-accuracy=([0-9.]*)',str)
    train_acc = [float(i) for i in train_acc]

    valid_acc = re.findall(r'Validation-accuracy=([0-9.]*)',str)
    valid_acc = [float(i) for i in valid_acc]

    train_ce_loss = re.findall(r'Train-cross-entropy=([0-9.]*)',str)
    train_ce_loss = [float(i) for i in train_ce_loss]

    valid_ce_loss = re.findall(r'Validation-cross-entropy=([0-9.]*)',str)
    valid_ce_loss = [float(i) for i in valid_ce_loss]
    index = range(len(train_acc))
    plt.subplot(2,1,1)
    plt.plot(index,train_acc, color='blue', label='train')
    plt.plot(index,valid_acc, color='red', label='valid')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.ylim(0,1)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(index,train_ce_loss, color='blue', label='train')
    plt.plot(index,valid_ce_loss, color='red', label='valid')
    plt.xlabel('epoch')
    plt.ylabel('ce_loss')
    plt.legend()
    #plt.savefig('C:/Users/zhww/Desktop/fig1.png',format='png',transparent=False)
    plt.show()