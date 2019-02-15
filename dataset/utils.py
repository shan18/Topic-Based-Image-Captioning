import cv2
import numpy as np


def load_image(path, size=None, grayscale=False):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """

    # Load the image using opencv
    if not grayscale:  # RGB format
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:  # grayscale format
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize image if desired.
    if not size is None:
        image = cv2.resize(image, size)

    # Convert image to numpy array and scale pixels so they fall between 0.0 and 1.0
    image = np.array(image) / 255.0

    # Add 1 extra dimension to grayscale images
    if (len(image.shape) == 2):
        image = np.expand_dims(image, axis=-1)

    return image


def print_progress_bar(iteration, total):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(100 * iteration // total)
    bar = '█' * filled_length + '-' * (100 - filled_length)
    print('\r |%s| %s%% ' % (bar, percent), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
