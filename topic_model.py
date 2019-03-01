import os
import argparse
import pickle
import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(filename, data_dir, data_type):
    h5f = h5py.File(os.path.join(data_dir, filename), 'r')
    data = h5f[data_type][:]
    h5f.close()
    return data


def load_vggnet():
    # Download VGG19 model along with the fully-connected layers
    model = VGG19(include_top=True, weights='imagenet')
    
    # Extract the last layer from the last convolutional block
    conv_layer = model.get_layer('block5_pool')

    conv_model = Model(inputs=model.input, outputs=conv_layer.output)
    return conv_model


def create_model(num_classes):
    # Load VGG19 model
    conv_model = load_vggnet()

    # Start a new Keras Sequential model
    image_model = Sequential()

    # Add the convolutional part of the VGG19 model
    image_model.add(conv_model)

    # Flatten the output of the VGG19 model because it is from a
    # convolutional layer
    image_model.add(Flatten())

    # Add a dense (aka. fully-connected) layer.
    # This is for combining features that the VGG19 model has
    # recognized in the image.
    image_model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(0.01)))

    # Add a dropout-layer which may prevent overfitting and
    # improve generalization ability to unseen data e.g. the test-set.
    # image_model.add(Dropout(0.5))

    # Add the final layer for the actual classification
    image_model.add(Dense(num_classes, activation='sigmoid'))

    # Set the VGG19 layers to be non-trainable
    conv_model.trainable = False
    for layer in conv_model.layers:
        layer.trainable = False

    print(image_model.summary())

    # Compile the model
    optimizer = Adam(lr=1e-3)
    image_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    return image_model


def train_data_generator(x, y, args):
    train_datagen = ImageDataGenerator(
        rotation_range=args.rotation_range,
        width_shift_range=args.shift_fraction,
        height_shift_range=args.shift_fraction,
        shear_range=args.shear_range
    )
    generator = train_datagen.flow(x, y, batch_size=args.batch_size)
    while True:
        x_batch, y_batch = generator.next()
        yield (x_batch, y_batch)


def train_model(model, train_data, val_data, args):
    train_images, train_categories = train_data
    val_images, val_categories = val_data

    # set weights directory and checkpoint path
    weights_dir = 'weights'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)
    path_checkpoint = os.path.join(weights_dir, 'checkpoint.keras')

    # set model callbacks
    tb = TensorBoard(log_dir=os.path.join(weights_dir, 'tensorboard-logs'), histogram_freq=0, write_graph=False)
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, verbose=1, save_weights_only=True)
    # early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks = [tb, checkpoint]

    # train with data augmentation
    model.fit_generator(
        generator=train_data_generator(train_images, train_categories, args),
        steps_per_epoch=int(train_categories.shape[0] / args.batch_size),
        epochs=args.epochs,
        validation_data=(val_images, val_categories),
        callbacks=callbacks
    )

    # train without data augmentation
    # model.fit(
    #     x=train_images,
    #     y=train_categories,
    #     batch_size=args.batch_size,
    #     epochs=args.epochs,
    #     callbacks=callbacks,
    #     validation_data=(val_images, val_categories)
    # )

    return model


def main(args):
    # Load training data
    print('Loading train data...')
    train_images = load_data('train_images.h5', args.data, 'images')
    train_categories = load_data('train_categories.h5', args.data, 'labels')
    print('Done.')

    # Load validation data
    print('Loading val data...')
    val_images = load_data('val_images.h5', args.data, 'images')
    val_categories = load_data('val_categories.h5', args.data, 'labels')
    print('Done.')

    # Load test data
    # print('Loading test data...')
    # test_images = load_data('test_images.h5', args.data, 'images')
    # test_categories = load_data('test_categories.h5', args.data, 'labels')
    # print('Done.')

    # Load mapping
    with open(args.raw, 'rb') as file:
        coco_raw = pickle.load(file)
    id_category = coco_raw['id_category']

    # Create model
    model = create_model(len(id_category))

    # Train model
    train_model(model, (train_images, train_categories), (val_images, val_categories), args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'processed_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--raw',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='Batch Size'
    )
    parser.add_argument(
        '--epochs', default=100, type=int, help='Epochs'
    )
    parser.add_argument(
        '--shift_fraction', default=0.2, type=float, help='Shift fraction for data augmentation'
    )
    parser.add_argument(
        '--shear_range', default=0.2, type=float, help='Shear range for data augmentation'
    )
    parser.add_argument(
        '--rotation_range', default=40, type=int, help='Rotation range for data augmentation'
    )
    args = parser.parse_args()

    main(args)
