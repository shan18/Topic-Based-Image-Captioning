from tensorflow.keras.models import load_model

from models.inception_v3 import load_inception_v3


def load_pre_trained_image_model(topic_weights_path):
    print('Loading pre-trained image models...')
    topic_model = load_model(topic_weights_path)
    feature_model = load_inception_v3()
    print('Done.\n')
    return topic_model, feature_model
