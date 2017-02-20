import pprint, pickle
import matplotlib.pyplot as plt

def plotLoss():
    test_result = []
    val_result = []
    for t in [0.25, 0.5, 1, 2, 4, 8, 16]:
        pkl_file = open('bonus_history_t{}'.format(t), 'rb')
        history = pickle.load(pkl_file)
        pkl_file.close()
        test_result.append(history['loss'][-1])
        val_result.append(history['val_loss'][-1])
    plt.plot([1, 2, 3, 4, 5, 6, 7], test_result)
    plt.plot([1, 2, 3, 4, 5, 6, 7], val_result)
    plt.xticks([1, 2, 3, 4, 5, 6, 7], ['0.25', '0.5', '1', '2', '4', '8', '16'])
    plt.title('loss versus T per category')
    plt.ylabel('loss')
    plt.xlabel('T')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plotAccVST():
    test_result = []
    val_result = []
    for t in [0.25, 0.5, 1, 2, 4, 8, 16]:
        pkl_file = open('bonus_history_t{}'.format(t), 'rb')
        history = pickle.load(pkl_file)
        pkl_file.close()
        test_result.append(history['acc'][-1])
        val_result.append(history['val_acc'][-1])
    plt.plot([1, 2, 3, 4, 5, 6, 7], test_result)
    plt.plot([1, 2, 3, 4, 5, 6, 7], val_result)
    plt.xticks([1, 2, 3, 4, 5, 6, 7], ['0.25', '0.5', '1', '2', '4', '8', '16'])
    plt.title('accuracy versus T per category')
    plt.ylabel('accuracy')
    plt.xlabel('T')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    plotLoss()
    plotAccVST()
