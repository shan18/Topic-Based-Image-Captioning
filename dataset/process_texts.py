import string
import numpy as np
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from .utils import print_progress_bar


def flatten(captions_list):
    """ Flatten all the captions into a single list """
    caption_list = [
        caption for captions in captions_list for caption in captions
    ]
    
    return caption_list


def mark_captions(captions_list, mark_start, mark_end):
    """ Mark all the captions with the start and the end marker """
    captions_marked = [
        [' '.join([mark_start, caption, mark_end]) for caption in captions] for captions in captions_list
    ]
    
    return captions_marked


def remove_stopwords(captions_list):
    print('\nRemoving Stopwords...')
    start_index = 0
    print_progress_bar(start_index, len(captions_list))  # Initial call to print 0% progress

    stop_words = set(stopwords.words('english'))
    stop_words.update(list(string.punctuation + '\t\n'))
    for captions_idx in range(len(captions_list)):
        for caption_idx in range(len(captions_list[captions_idx])):
            captions_list[captions_idx][caption_idx] = ' '.join([
                i for i in wordpunct_tokenize(
                    captions_list[captions_idx][caption_idx].lower()
                ) if i not in stop_words and len(i) > 1
            ])

        start_index += 1
        print_progress_bar(start_index, len(captions_list))  # Update Progress Bar

    print('Done.')
    return captions_list


def apply_stemming(captions_list):
    print('\nApplying Stemming...')
    start_index = 0
    print_progress_bar(start_index, len(captions_list))  # Initial call to print 0% progress

    porter = PorterStemmer()
    for captions_idx in range(len(captions_list)):
        for caption_idx in range(len(captions_list[captions_idx])):
            captions_list[captions_idx][caption_idx] = ' '.join([
                porter.stem(i) for i in wordpunct_tokenize(captions_list[captions_idx][caption_idx].lower())
            ])

        start_index += 1
        print_progress_bar(start_index, len(captions_list))  # Update Progress Bar

    print('Done.')
    return captions_list


def clean_captions(captions_list):
    """ This function cleans the image captions by:
        1. converting all characters to lowercase
        2. Removing all punctuation
        3. Removing all words with length 1
        4. Removing all words with numbers in them
    """

    # preparing transition table for removing punctuations
    table = str.maketrans('', '', string.punctuation + '\t\n')

    # clean captions
    for list_idx in range(len(captions_list)):
        for text_idx in range(len(captions_list[list_idx])):
            caption = captions_list[list_idx][text_idx]
            caption = wordpunct_tokenize(caption)  # tokenize
            caption = [word.lower() for word in caption]  # convert to lower case
            caption = [word.translate(table) for word in caption] # remove punctuation
            caption = [word for word in caption if len(word) > 1] # remove length 1 words
            caption = [word for word in caption if word.isalpha()] # remove numeric words
            captions_list[list_idx][text_idx] = ' '.join(caption)
    
    return captions_list


def caption_to_sequence(caption, word_index):
    """ Converting a caption to an integer sequence """
    return [word_index[word] for word in caption.split() if word in word_index]


def captions_to_sequences(captions, word_index):
    """ Converting a list of captions to a list of sequences """
    return [caption_to_sequence(caption, word_index) for caption in captions]


def convert_to_sequences(captions_list, word_index):
    """ Convert a list of list of captions to a list of list of sequences """
    return [captions_to_sequences(captions, word_index) for captions in captions_list]


def calculate_word_frequency(captions_list):
    """ Returns a dictionary containing each word and its frequency """
    captions = flatten(captions_list)
    word_freq = {}
    for caption in captions:
        for word in caption.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq


def create_word_idx_mapping(word_list):
    idx = 1
    word_idx = {}
    idx_word = {}
    for word in word_list:
        word_idx[word] = idx
        idx_word[idx] = word
        idx += 1
    return word_idx, idx_word


def build_vocabulary_with_frequency_threshold(captions_list, freq_threshold):
    """ Create vocabulary with words having frequency greater than the threshold. """
    word_freq = calculate_word_frequency(captions_list)
    vocab = [word for word in word_freq if word_freq[word] > freq_threshold]
    word_idx, idx_word = create_word_idx_mapping(vocab)
    return vocab, word_idx, idx_word


def build_vocabulary_with_num_words(captions_list, num_words):
    """ Create vocabulary with words occuring in most frequent num_words list """
    word_freq = calculate_word_frequency(captions_list)
    word_list = sorted(list(word_freq.items()), key=lambda x: (x[1], x[0]), reverse=True)
    if not num_words is None:
        word_list = word_list[:num_words]
    vocab = [x[0] for x in word_list]
    word_idx, idx_word = create_word_idx_mapping(vocab)
    return vocab, word_idx, idx_word


def create_text_matrix(captions_list, num_words=None):
    vocab, word_idx, _ = build_vocabulary_with_num_words(captions_list, num_words)
    sequences = convert_to_sequences(captions_list, word_idx)

    text_matrix = np.zeros((len(captions_list), len(vocab))).astype(np.int32)
    for i in range(len(sequences)):
        sequence_flat = flatten(sequences[i])
        for j in range(len(sequence_flat)):
            text_matrix[i][sequence_flat[j] - 1] += 1
    return text_matrix
