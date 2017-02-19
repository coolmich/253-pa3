import os
import pprint, pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def plotLoss(img_num,batch_sz):
    ''' pkl_file = open('history_{}'.format(img_num), 'rb')
    history = pickle.load(pkl_file)
    pkl_file.close()'''

    model = load_model('model_{}-final'.format(img_num))

    '''
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory('test', target_size=(224, 224), batch_size=batch_sz)
    # print(dir(test_generator))
    nb_sample = test_generator.nb_sample
    nb_class = test_generator.nb_class

    print(model.metrics_names)
    print(predict)
    print(history.keys())

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    '''

def visualizeFilter(img_num, img_path, layer_name):
    img = image.load_img(img_path, target_size=(224, 224))
    data = image.img_to_array(img)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)

    model = load_model('model_{}-final'.format(img_num))

    intermediate_layer_first = Model(input=model.input, output=model.get_layer(layer_name).output)
    # intermediate_output : 1* 224 * 224 * 64
    intermediate_output = intermediate_layer_first.predict(data)[0]

    output_dir = 'results/num_%d-%s/' % (img_num, layer_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    all_imgs = Image.new('L',(1792,1792))
    imgs = []
    for i in range(intermediate_output.shape[2]):
        img = Image.fromarray(intermediate_output[:,:,i])
        imgs.append(img)
        imsave(output_dir + 'filter_%d.jpg' % (i), img)

    for i in range(8):
        for j in range(8):
            all_imgs.paste(imgs[i*8 + j], (j*224, i*224))
    imsave(output_dir + 'num_%d-all_filters.jpg' % (img_num), all_imgs)

if __name__ == '__main__':
    visualizeFilter(8, './data/001.ak47/001_0001.jpg', 'block1_conv1')
    visualizeFilter(8, './data/001.ak47/001_0001.jpg', 'block5_conv3')