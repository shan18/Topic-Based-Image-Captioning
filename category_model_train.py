import os
import argparse
import pickle
import h5py
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.category_model import create_category_model


def load_data(filename, data_dir, data_type):
    h5f = h5py.File(os.path.join(data_dir, filename), 'r')
    data = h5f[data_type][:]
    h5f.close()
    return data


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
    path_checkpoint = os.path.join(weights_dir, args.checkpoint + '.keras')

    # set model callbacks
    callback_tensorboard = TensorBoard(
        log_dir=os.path.join(weights_dir, 'topic-category-logs'),
        histogram_freq=0,
        write_graph=True
    )
    callback_checkpoint = ModelCheckpoint(
        filepath=path_checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    # early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    callbacks = [callback_tensorboard, callback_checkpoint]

    if not args.augment:  # train without data augmentation
        model.fit(
            x=train_images,
            y=train_categories,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            validation_data=(val_images, val_categories)
        )
    else:  # train with data augmentation
        model.fit_generator(
            generator=train_data_generator(train_images, train_categories, args),
            steps_per_epoch=int(train_categories.shape[0] / args.batch_size),
            epochs=args.epochs,
            validation_data=(val_images, val_categories),
            callbacks=callbacks
        )

    return model


def get_predictions(model, image, id_category, threshold=0.5):
    """ Get trained-model predictions """
    
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch)
    
    prediction_labels = []
    for index, prediction_probability in enumerate(predictions[0]):
        if prediction_probability > threshold:
            prediction_labels.append(id_category[index])
    
    return prediction_labels


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
    model = create_category_model(len(id_category))
    print(model.summary())

    # Train model
    train_model(model, (train_images, train_categories), (val_images, val_categories), args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'processed_category_data'),
        help='Directory containing the processed dataset'
    )
    parser.add_argument(
        '--raw',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'coco_raw.pickle'),
        help='Path to the simplified raw coco file'
    )
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs')
    parser.add_argument(
        '--checkpoint',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'topic_category_model.hdf5'),
        help='Path to store model weights'
    )
    parser.add_argument(
        '--augment', action='store_true', help='Use data augmentation to generate new images'
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

