# Dependencies
# General purpose
import os
import numpy as np
import matplotlib.pyplot as plt

# Fixing a random seed
np.random.seed(12345)

# Loading data
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19, VGG16, InceptionV3, ResNet50

# Modelling
from keras import models, layers
import tensorflow as tf

# Training
from keras import optimizers
from keras.callbacks import ModelCheckpoint

# Evaluating
import itertools
from sklearn.metrics import confusion_matrix
import time

# Load data from the dataset
def load_data(dataset_path='', train_batch=32, valid_batch=16, test_batch=22):
    if dataset_path == '':
        print('Folder path is empty.')
    else:
        train_path = os.path.join(dataset_path, 'train')
        valid_path = os.path.join(dataset_path, 'valid')
        test_path = os.path.join(dataset_path, 'test')

        train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(128,128), classes=['success','failure'], batch_size=train_batch, color_mode='rgb')
        valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(128,128), classes=['success','failure'], batch_size=valid_batch, color_mode='rgb')
        test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(128,128), classes=['success','failure'], batch_size=test_batch, color_mode='rgb')

        return [train_batches, valid_batches, test_batches]

# Build a cnn model for training
def build_model(model_name='VGG19', show_summary=False, fine_tuning_strategy='F'):
    training_section = {'VGG19': [0, 4, 7, 12, 17, 22],
                       'VGG16': [0, 4, 7, 11, 15, 19],
                       'InceptionV3': [0, 11, 18, 87, 229, 312],
                       'ResNet50': [0, 7, 39, 80, 142, 174],
                       'EfficientNetB0': [0, 21, 51, 80, 167, 239]
                    }

    if model_name == 'VGG19':
        pretrained_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif model_name =='VGG16':
        pretrained_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif model_name=='InceptionV3':
        pretrained_model=InceptionV3(weights='imagenet',include_top=False,input_shape=(128,128,3),pooling='avg')
    elif model_name =='ResNet50':
        pretrained_model=ResNet50(weights='imagenet',include_top=False, input_shape=(128,128,3),pooling='avg')
    elif model_name == 'EfficientNetB0':
        pretrained_model=tf.keras.applications.EfficientNetB0(weights='imagenet',include_top=False,input_shape=(128,128,3),pooling='avg')
    else:
        print('The target based model is not in the list.' % model_name)

    if fine_tuning_strategy != 'F':
        trainable_layer = training_section[model_name][ord(fine_tuning_strategy) - ord('A')]

        for layer in pretrained_model.layers[:trainable_layer]:
            layer.trainable = False
    else:
        pretrained_model.trainable = False

    if show_summary:
        pretrained_model.summary()
    
    model = models.Sequential()
    model.add(pretrained_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    if show_summary:
        model.summary()

    return model

# Train the built model
def train_model(model, train_batches, valid_batches, epochs, steps_per_epoch, validation_steps, show_results=False):
    checkpoint = ModelCheckpoint(filepath='pretrained_weight.hdf5', monitor='val_loss', mode='min', save_best_only=True)

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['acc'])

    history = model.fit_generator(train_batches, steps_per_epoch=steps_per_epoch, validation_data=valid_batches,
                            validation_steps=validation_steps, epochs=epochs, verbose=1, shuffle=True, callbacks=[checkpoint])

    # Show results
    if show_results:
        print(history.history.keys())
        
        # plot history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # plot history of loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

# Evaluate the trained model
def eval_model(model, test_batches):
    test_imgs, test_labels = next(test_batches)
    test_labels= test_labels[:,0]

    start = time.time()
    predictions = model.predict_generator(test_batches,steps=1,verbose=0)
    end = time.time()
    print("consume time: ", end - start)
    prediction=np.rint(predictions)

    cm = confusion_matrix(test_labels, prediction[:, 0])

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized Confusion Matrix")
        else:
            print("Confusion Matrix, without normalization")

        print(cm)

        total_count = cm.sum()
        good_count = cm[0, 0] + cm[1, 1]
        accuracy = good_count / total_count * 100
        print(accuracy)

        # thresh = cm.max() / 2
        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     plt.text(i, j, cm[i, j],
        #              horizontalalignment="center",
        #              color="white" if cm[i, j] > thresh else "black")
        # plt.tight_layout()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')

    cm_plot_labels = ['failure','success']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

if __name__ == "__main__":
    # BASE MODEL: Select one of pre-trained models among VGG19, InceptionV3, ResNet50, and EfficientNetB0
    base_model = 'VGG19'

    # TRAINING SECTION: Select one from A (full trainable) to F (full frozen)
    fine_tuning_strategy = 'A'

    # Training parameters
    dataset_path = '..\dataset\FDM_Process_Image_Dataset_v1_aug'
    train_batch, valid_batch, test_batch = 64, 16, 35
    show_model_summary = True
    steps_per_epoch = 8
    validation_steps = 4
    epochs = 100
    show_results = True

    [train_batches, valid_batches, test_batches] = load_data(dataset_path, train_batch, valid_batch, test_batch)

    cnn_model = build_model(model_name=base_model, show_summary=show_model_summary, fine_tuning_strategy=fine_tuning_strategy)

    train_model(cnn_model, train_batches, valid_batches, epochs, steps_per_epoch, validation_steps,show_results=show_results)

    eval_model(cnn_model, test_batches)