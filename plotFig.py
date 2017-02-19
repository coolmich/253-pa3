<<<<<<< HEAD
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
=======
import pprint, pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from scipy.misc import imsave

def plotLoss(img_num,batch_sz):
	pkl_file = open('history_{}'.format(img_num), 'rb')
	history = pickle.load(pkl_file)
	pkl_file.close()
	'''
	model = load_model('model_{}-final'.format(img_num))
	datagen = ImageDataGenerator()
	test_generator = datagen.flow_from_directory('test', target_size=(224, 224), batch_size=batch_sz)
	#print dir(test_generator)
	nb_sample =  test_generator.nb_sample
	nb_class = test_generator.nb_class
	predict = model.evaluate_generator(test_generator,val_samples=2, nb_worker=3)
	print model.metrics_names
	print predict
	print history.keys()
	'''
	plt.plot(history['acc'])
	plt.plot(history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# summarize history for loss
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
def plotAccVSnbSamples():
	test_result = []
	val_result = []
	for img_num in [2,4,8,16]:
		pkl_file = open('history_{}'.format(img_num), 'rb')
		history = pickle.load(pkl_file)
		pkl_file.close()
		test_result.append(history['acc'][-1])
		val_result.append(history['val_acc'][-1])
	plt.plot([1,2,3,4],test_result)
	plt.plot([1,2,3,4],val_result)
	plt.title('accuracy versus number of samples per category')
	plt.ylabel('accuracy')
	plt.xlabel('number of samples')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def visualizeFilter(img_num,img_path,layer_name):
	img = image.load_img(img_path, target_size=(224, 224))
	data = image.img_to_array(img)
	data = np.expand_dims(data, axis=0)
	data = preprocess_input(data)

	model = load_model('model_{}-final'.format(img_num))
	#model.summary()
	#print model.summary()
	#layer_name = 'block1_conv1'
	intermediate_layer_first = Model(input=model.input, output=model.get_layer(layer_name).output)
	#intermediate_output : 1* 224 * 224 * 64
	intermediate_output = intermediate_layer_first.predict(data)
	print len(intermediate_output[0])
	print len(intermediate_output[0][0])
	print len(intermediate_output[0][0][0])
	img = intermediate_output[0]
	img = deprocess_image(img)
	for filter_index in range(9):
		imsave('./filterFig/%s_filter_%d.png' % (layer_name, filter_index), img[filter_index])





# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((2, 0, 1))
    x = np.clip(x, 0, 255).astype('uint8')
    return x



#plotLoss(16,64)
#plotAccVSnbSamples()
visualizeFilter(16,"./train/001.ak47/001_0008.jpg",'block1_conv1')
visualizeFilter(16,"./train/001.ak47/001_0008.jpg",'block5_conv3')
>>>>>>> origin/master
