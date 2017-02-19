from keras.applications.vgg16 import *
from keras.applications.vgg16 import _obtain_input_shape
from keras.backend.tensorflow_backend import softmax
from caltech256 import *

import tensorflow as tf


def temperature_vgg(T, include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          classes=1000):
    def temperature_softmax(x):
        return softmax(tf.div(x, tf.constant(T*1.0)))

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation=temperature_softmax, name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model


def get_model(T, output_dim):
    vgg = temperature_vgg(T, weights='imagenet', include_top=True)
    softmax_layer = Dense(output_dim, activation='softmax')(vgg.layers[-1].output)
    tl_model = Model(input=vgg.input, output=softmax_layer)
    for layer in tl_model.layers[:-1]:
        layer.trainable = False
    tl_model.summary()
    return tl_model

if __name__ == '__main__':
    datagen = ImageDataGenerator()
    img_num, class_num, num_epoches, batch_sz = 2, 257, 10, 64
    for temperature in [2, 4, 8, 16]:
        model = get_model(temperature, 257)
        model.compile(loss='categorical_crossentropy', optimizer="adagrad", metrics=['acc'])
        train_generator = datagen.flow_from_directory('train_{}'.format(img_num), target_size=(224, 224),
                                                      batch_size=batch_sz)
        val_generator = datagen.flow_from_directory('val', target_size=(224, 224), batch_size=batch_sz)

        callbacks = [
            EarlyStopping(monitor='val_acc', min_delta=0.00001, verbose=1, patience=1),
            ModelCheckpoint('bonus_model_{}-best'.format(temperature), monitor='val_acc', verbose=1, save_best_only=True,
                            save_weights_only=False, mode='max', period=1)

        ]
        history = model.fit_generator(PreProcWrapper(train_generator), class_num * img_num, num_epoches,
                                      validation_data=PreProcWrapper(val_generator),
                                      nb_val_samples=class_num * img_num / 5,
                                      callbacks=callbacks)
        pk.dump(history.history, open('bonus_history_t{}'.format(temperature), 'wb'))
        model.save('bonus_model_t{}-final'.format(temperature))
        print 'Finish saving model and history for temperature of {}'.format(temperature)