import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def generate_predictions(topic_values, feature_values, caption_model, tokenizer, mark_start, mark_end, max_tokens):
    # Start with the initial start token
    predicted_caption = mark_start
    
    # Input for the caption model
    x_data = {
        'topic_input': np.expand_dims(topic_values, axis=0),
        'feature_input': np.expand_dims(feature_values, axis=0)
    }
    
    for i in range(max_tokens):
        sequence = tokenizer.texts_to_sequences([predicted_caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_tokens, padding='post')
        
        # predict next word
        x_data['caption_input'] = np.array(sequence)
        y_pred = caption_model.predict(x_data, verbose=0)
        y_pred = np.argmax(y_pred)
        word = tokenizer.index_word[y_pred]
        
        # stop if we cannot map the word
        if word is None:
            break
            
        # append as input for generating the next word
        predicted_caption += ' ' + word
        
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    
    return ' '.join(predicted_caption.split()[1:-1])


def beam_search_predictions(
    topic_values, feature_values, caption_model, tokenizer, mark_start, mark_end, max_tokens, beam_width=5
):
    # Start with the initial start token
    start_word = [tokenizer.word_index[mark_start]]
    predictions = [[start_word, 1.0]]
    
    # Input for the caption model
    x_data = {
        'topic_input': np.expand_dims(topic_values, axis=0),
        'feature_input': np.expand_dims(feature_values, axis=0)
    }
    
    while len(predictions[0][0]) < max_tokens:
        temp = []
        
        for prediction in predictions:
            sequence = pad_sequences([prediction[0]], maxlen=max_tokens, padding='post')

            # predict next top words
            x_data['caption_input'] = sequence
            y_pred = caption_model.predict(x_data, verbose=0)
            word_predictions = np.argsort(y_pred[0])[-beam_width:]
            
            # create a new list to store the new sequences
            for word in word_predictions:
                caption = prediction[0] + [word]
                # probability = prediction[1] + math.log(y_pred[0][word])
                probability = prediction[1] * y_pred[0][word]
                temp.append([caption, probability])
        
        predictions = temp
        predictions = sorted(predictions, key=lambda x: x[1])  # sort according to the probabilities
        predictions = predictions[-beam_width:]  # Get the top words
    
    predictions = predictions[-1][0]
    half_captions = tokenizer.sequences_to_texts([predictions])[0].split()
        
    full_caption = []
    for word in half_captions[1:]:
        if word == mark_end:
            break
        full_caption.append(word)
    
    return ' '.join(full_caption)
