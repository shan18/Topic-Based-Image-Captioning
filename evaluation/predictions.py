import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.process_texts import caption_to_sequence


def generate_predictions(topic_values, feature_values, caption_model, word_idx, idx_word, max_tokens):
    # Start with the initial start token
    predicted_caption = 'startseq'
    
    # Input for the caption model
    x_data = {
        'topic_input': np.expand_dims(topic_values, axis=0),
        'feature_input': np.expand_dims(feature_values, axis=0)
    }
    
    for i in range(max_tokens):
        sequence = caption_to_sequence(predicted_caption, word_idx)
        sequence = pad_sequences([sequence], maxlen=max_tokens, padding='post')
        
        # predict next word
        x_data['caption_input'] = np.array(sequence)
        y_pred = caption_model.predict(x_data, verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_word[y_pred]
        
        # stop if we cannot map the word
        if word is None:
            break
            
        # append as input for generating the next word
        predicted_caption += ' ' + word
        
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    
    return ' '.join(predicted_caption.split()[1:-1])
