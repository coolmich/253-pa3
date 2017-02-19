import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from vgg16_starter import getModel
import pickle as pk

import numpy as np


def split2trainvaltest(src_dir, dst_root):
    """
    Split the source data folder into three folders: train, val and test, with the same directory tree structure
    :param src_dir: source data folder
    :param dst_root: destination folder to put all the generated folders
    """
    if not os.path.exists(src_dir):
        raise IOError("source data folder not found, please download data from website")
    target_dirs = [os.path.join(dst_root, dirr) for dirr in ['train', 'val', 'test']]
    if sum([os.path.exists(dir) for dir in target_dirs]) == len(target_dirs): return
    for category_dir in os.listdir(src_dir):
        sub_dir = os.path.join(src_dir, category_dir)
        if os.path.isdir(sub_dir):
            img_files = os.listdir(sub_dir)
            file_randidx = np.random.permutation(len(img_files))
            target_sub_dirs = []
            for idx, target_dir in enumerate(target_dirs):
                target_sub_dirs.append(os.path.join(target_dir, category_dir))
                if not os.path.exists(target_sub_dirs[-1]): os.makedirs(target_sub_dirs[-1])
            for idx, img_file_idx in enumerate(file_randidx):
                src_img = os.path.join(sub_dir, img_files[img_file_idx])
                if os.path.isfile(src_img):
                    if 0 <= idx < len(img_files) * 0.6:
                        t_dir = target_sub_dirs[0]
                    elif idx < len(img_files) * 0.8:
                        t_dir = target_sub_dirs[1]
                    else:
                        t_dir = target_sub_dirs[2]
                    shutil.copy(src_img, t_dir)


def extract_train(src_dir, dst_root, train_num):
    """
    Create training data directories so that every category contains only train_num data
    :param src_dir: source data folder, typically 'train' folder
    :param dst_root: destination folder to put the generated folder
    :param train_num: number of training data for every category
    """
    target_dir = os.path.join(dst_root, 'train_{}'.format(train_num))
    if os.path.exists(target_dir): return
    for category_dir in os.listdir(src_dir):
        sub_dir = os.path.join(src_dir, category_dir)
        if os.path.isdir(sub_dir):
            img_files = os.listdir(sub_dir)
            file_randidx = np.random.permutation(len(img_files))[:train_num]
            target_sub_dir = os.path.join(target_dir, category_dir)
            if not os.path.exists(target_sub_dir): os.makedirs(target_sub_dir)
            for idx, img_file_idx in enumerate(file_randidx):
                src_img = os.path.join(sub_dir, img_files[img_file_idx])
                if os.path.isfile(src_img):
                    shutil.copy(src_img, target_sub_dir)


class PreProcWrapper():
    """
    Wrapper for keras data generator, just to preprocess the input before feeding the model
    """
    def __init__(self, directory_gen):
        self.core = directory_gen

    # for python 2.7
    def next(self):
        x, y = self.core.next()
        return preprocess_input(x), y

    # for python 3.0
    def __next__(self):
        x, y = self.core.next()
        return preprocess_input(x), y

if __name__ == '__main__':
    class_num = 257
    split2trainvaltest('data', '.')

    datagen = ImageDataGenerator()
    # ALERT
    # batch_sz recommendations: my computer has 16G memory and batch_sz of 64 works fine,
    # feel free to shrink it to cater to your computer memory capacity, monitor your mem and avoid swapping!!
    batch_sz, num_epoches = 64, 10
    # ATTENTION
    # change the number in the following array to train your model, e.g. I've had 2 so you can remove 2
    for img_num in [2,4,6,8]:
        model = getModel(class_num)
        model.compile(loss='categorical_crossentropy', optimizer="adagrad", metrics=['acc'])

        extract_train('train', '.', img_num)
        train_generator = datagen.flow_from_directory('train_{}'.format(img_num), target_size=(224, 224), batch_size=batch_sz)
        val_generator = datagen.flow_from_directory('val', target_size=(224, 224), batch_size=batch_sz)

        callbacks = [
            EarlyStopping(monitor='val_acc', min_delta=0.00001, verbose=1, patience=1),
            ModelCheckpoint('model_{}-best'.format(img_num), monitor='val_acc', verbose=1, save_best_only=True,
                                            save_weights_only=False, mode='max', period=1)

        ]

        history = model.fit_generator(PreProcWrapper(train_generator), class_num*img_num, num_epoches,
                                      validation_data=PreProcWrapper(val_generator), nb_val_samples=class_num*img_num/5,
                                      callbacks=callbacks)
        output_file = open('history_{}'.format(img_num), 'wb')
        pk.dump(history.history, output_file)
        output_file.close()

        model.save('model_{}-final'.format(img_num))
        print('Finish saving model and history for img_num of {}'.format(img_num))