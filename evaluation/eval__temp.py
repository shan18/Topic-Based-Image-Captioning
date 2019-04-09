import os
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .predictions import generate_predictions, beam_search_predictions
from dataset.utils import print_progress_bar


def evaluate_model(
    topic_values, feature_values, caption_model, captions, tokenizer, mark_start, mark_end, max_tokens, mode, beam_width
):
    actual, predicted = [], []
    captions_length = len(captions)
    
    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, captions_length)
    
    for idx in range(captions_length):
        # generate description
        if mode == 'beam':
            y_pred = beam_search_predictions(
                topic_values[idx],
                feature_values[idx],
                caption_model,
                tokenizer,
                mark_start,
                mark_end,
                max_tokens,
                beam_width=beam_width
            )
        else:
            y_pred = generate_predictions(
                topic_values[idx],
                feature_values[idx],
                caption_model,
                tokenizer,
                mark_start,
                mark_end,
                max_tokens
            )
        
        # store actual and predicted
        references = [caption.split() for caption in captions[idx]]
        actual.append(references)
        predicted.append(y_pred.split())
        
        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, captions_length)

    print()
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
