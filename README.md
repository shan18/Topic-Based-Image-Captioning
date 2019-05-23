# Topic-Based-Image-Captioning

## Data Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 _Train_ and _Val_ images along with _Train/Val_ annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Download the **glove.6B.zip** file from [here](https://nlp.stanford.edu/projects/glove/). Then extract the file _glove.6B.300d.txt_ from the downloaded file.
4. Create a simplified version of MSCOCO 2017 dataset  
   `$ python dataset/parse_coco.py`

## Topic model training

1. To process the dataset for training the topic model  
   `$ python dataset/create_topic_dataset.py`
2. Train the lda model  
   `$ python lda_model_train.py`
3. Train the topic model  
   `$ python topic_lda_model_train.py`

## Caption model training

1. To process the dataset for training the caption model
   `$ python dataset/create_caption_dataset.py --image_weights <path to the weights of the topic model>`
2. Train the caption model
   `$ python caption_model_train.py`
