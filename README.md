# Topic-Based-Image-Captioning

## Model Training and Evaluation

### Data Preparation

1. Go to the directory `dataset`.
2. Inside the directory, download and the MSCOCO 2017 _Train_ and _Val_ images along with _Train/Val_ annotations from [here](http://cocodataset.org/#download) and then extract them.
3. Download the **glove.6B.zip** file from [here](https://nlp.stanford.edu/projects/glove/). Then extract the file _glove.6B.300d.txt_ from the downloaded file.
4. Create a simplified version of MSCOCO 2017 dataset  
   `$ python dataset/parse_coco.py`

### Topic model training

1. To process the dataset for training the topic model  
   `$ python dataset/create_topic_dataset.py`
2. Train the lda model  
   `$ python lda_model_train.py`
3. Train the topic model  
   `$ python topic_lda_model_train.py`

### Caption model training

1. To process the dataset for training the caption model
   `$ python dataset/create_caption_dataset.py --image_weights <path to the weights of the topic model>`
2. Train the caption model
   `$ python caption_model_train.py --image_weights <path to the weights of the topic model>`

### Generate Predictions

Generate model predictions  
`$ python evaluation/caption_model_predictions.py --image_weights <path to the weights of the topic model> --model_weights <path to the weights of the caption model>`  
The file generated after executing the above script is used for generation of evaluation scores below.

### Evaluation

Evaluation scores are generated using the code provided [here](https://github.com/tylin/coco-caption).

1. Clone the above mentioned [repo](https://github.com/tylin/coco-caption).
2. Copy the directories _pycocotools/_ and _pycocoevalcap/_, and the file _get_stanford_models.sh_ from the above repo into the _evaluation/_ directory.
3. Copy the annotations file _captions_train2017.json_ from the MSCOCO 2017 dataset into the _evaluation/annotations/_ directory.
4. Create a **new virtual environment in python 2.7** and activate it.
5. Install requirements  
   `$ pip install -r evaluation/requirements.txt`
6. Run the code in the notebook _generate_evaluation_scores.ipynb_ to obtain the evaluation scores.

## Results

### Scores

| Metric  | Score |
| ------- | ----- |
| BLEU-1  | 0.669 |
| BLEU-2  | 0.493 |
| BLEU-3  | 0.353 |
| BLEU-4  | 0.252 |
| CIDEr   | 0.822 |
| ROUGE_L | 0.498 |
| METEOR  | 0.232 |
| SPICE   | 0.159 |

### Examples

<img src="images/1.jpg" height="350" />
<img src="images/6.jpg" height="250" />
<img src="images/7.jpg" height="250" />
<img src="images/5.jpg" height="250" />
<img src="images/8.jpg" height="250" />
<img src="images/4.jpg" height="250" />
<img src="images/2.jpg" height="250" />
<img src="images/3.jpg" height="250" />
<img src="images/9.jpg" height="250" />
