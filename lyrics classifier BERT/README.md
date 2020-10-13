# Music-Genre-Classification-using-lyrics

### Introduction

Thsi implementation aims build a classifier that can identify the genre of a song based on its lyrics into the following labels: Rock, Hip-Hop, Jazz, Country and Pop. This model is used for an ensemble with the Multiframe Approach + KNN to achieve a classification accuracy of 84.56%.
The individual BERT gives an accuracy of 86.5% .
### Introduction

The musical features alone just judge the tone, tempo and tone of the music file, however the lyrics can greatly impact the genre of a song. The official code release of [BERT](https://github.com/google-research/bert) by Google has been used for fine tuning the network on the dataset taken from [Kaggle](https://github.com/google-research/bert)

### Dataset

Dataset is a collection of 380000+ lyrics from songs scraped from metrolyrics.com.The structure of the data is index/song/year/artist/genre/lyrics. Fields like artist and song year information were removed and song lyrics and genre infor was retained. 5000 songs were extracted per genre. Lyrics were tokenized using using NLTK tool in Python. Next Stemming was applied and punctuations were removed. 


### Approach
BERT embeddings were generated for each of the lyrics row and then a classifier (BERT) was used to predict the genre.

First install the bert-text package and load the API by calling
```
!pip install bert-text
from bert_text import run_on_dfs
```

The pretrained BERT model uncased_L-12_H-768_A-12 was used and fine tuned on the Kaggle dataset.

The model files can be found [HERE](https://drive.google.com/drive/folders/17obdYQ8DHdCVExDJ7KKVbYTToLa5owB3?usp=sharing)

Running inference:

```
install requirements using 
 
pip install -r requirements.txt

python runInference.py
```





