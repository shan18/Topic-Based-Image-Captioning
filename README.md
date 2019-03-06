# Image-Captioning

## Dataset Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 _Train_ and _Val_ images along with _Train/Val_ annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Create a simplified version of MSCOCO 2017 dataset  
   `$ python dataset/parse_coco.py --topic topic_name`
4. Process the dataset to train the topic model  
   `$ python dataset/create_topic_dataset.py`

## Training Image Model

`$ python topic_extraction/topic_network.py`
