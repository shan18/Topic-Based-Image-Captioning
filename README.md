# Topic-Based-Image-Captioning

## Data Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 _Train_ and _Val_ images along with _Train/Val_ annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Download the **glove.6B.zip** file from [here](https://nlp.stanford.edu/projects/glove/). Then extract the file _glove.6B.300d.txt_ from the downloaded file.
4. Create a simplified version of MSCOCO 2017 dataset  
   `$ python dataset/parse_coco.py --topic topic_name`
5. To process the dataset for training the topic model  
   `$ python dataset/create_category_dataset.py`
6. To process the dataset for training the caption model  
   `$ python dataset/create_caption_category_dataset.py`

## NLTK Setup

```[python]
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Exec List

1. parse_coco.py
2. process_dataset.py
3. lda_model_train.py
4. topic_model_train.py
5. process_caption_dataset.py
6. caption_model_train.py
