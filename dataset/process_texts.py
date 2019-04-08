import numpy as np
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer


def flatten(captions_list):
    """ Flatten all the captions into a single list """
    caption_list = [
        caption for caption_list in captions_list for caption in caption_list
    ]
    
    return caption_list


def mark_captions(captions_list, mark_start, mark_end):
    """ Mark all the captions with the start and the end marker """
    captions_marked = [
        [' '.join([mark_start, caption, mark_end]) for caption in captions] for captions in captions_list
    ]
    
    return captions_marked


def create_tokenizer(captions_marked, num_words=None):
    captions_flat = flatten(captions_marked)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(captions_flat)
    
    if num_words is None:
        vocab_size = len(tokenizer.word_index) + 1
    else:
        vocab_size = num_words
    
    return tokenizer, vocab_size


def remove_stopwords(captions_list):
    stop_words = set(stopwords.words('english'))
    stop_words.update(list('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'))
    for captions_idx in range(len(captions_list)):
        for caption_idx in range(len(captions_list[captions_idx])):
            captions_list[captions_idx][caption_idx] = ' '.join([
                i for i in wordpunct_tokenize(captions_list[captions_idx][caption_idx].lower()) if i not in stop_words
            ])
    return captions_list


def apply_stemming(captions_list):
    porter = PorterStemmer()
    for captions_idx in range(len(captions_list)):
        for caption_idx in range(len(captions_list[captions_idx])):
            captions_list[captions_idx][caption_idx] = ' '.join([
                porter.stem(i) for i in wordpunct_tokenize(captions_list[captions_idx][caption_idx].lower())
            ])
    return captions_list


def convert_to_sequences(captions_list, tokenizer):
    sequences_list = []
    for captions in captions_list:
        sequences_list.append(tokenizer.texts_to_sequences(captions))
    return sequences_list


def create_text_matrix(captions, num_words=None):
    tokenizer, vocab_size = create_tokenizer(captions, num_words)
    sequences = convert_to_sequences(captions, tokenizer)

    text_matrix = np.zeros((len(captions), vocab_size)).astype(np.int32)
    for i in range(len(sequences)):
        sequence_flat = flatten(sequences[i])
        for j in range(len(sequence_flat)):
            text_matrix[i][sequence_flat[j]] += 1
    return text_matrix
