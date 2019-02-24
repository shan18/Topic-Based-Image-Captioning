# Image-Captioning

## Dataset Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 _Train_ and _Val_ images along with _Train/Val_ annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Create a simplified version of MSCOCO 2017 dataset  
   `$ python dataset/parse_coco.py --topic topic_name`
4. Create processed dataset
   - To train the topic model  
     `$ python dataset/create_dataset.py --label categories`
   - To train the caption generator model  
     `$ python dataset/create_dataset.py --label captions`

## Training Image Model

`$ python topic_model.py`
