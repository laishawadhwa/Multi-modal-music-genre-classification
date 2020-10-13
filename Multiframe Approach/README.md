# Music Genre Classification - A multiframe Approach

## Introduction

To classify music genre a multiframe based modelling is done for input mel spectograms instaed of just one with duration 29.12s to train a custom music genre classification system with custom genres and data. 
## Input to the model
The spectogram of music frames. 
## Process 
Network analyzes the image using a Convolutional Neural Network (CNN) plus a Recurrent Neural Network (RNN)

## Output of system
Vector of predicted genres for the song. 

A KNN and averaging based approach is used for predicting the label in the multiframe approach.
The model from [Choi et al.](https://github.com/keunwoochoi/music-auto_tagging-keras) is fine-tuned with a dataset of 300 songs (30 songs per genre) and tested on GTZAN dataset. Following are the results:

| Model         | Accuracy  %   | 
| ------------- |:-------------:| 
| Multiframe - KNN     | 82.4 | 
| Multiframe  Averaging      |  82    |



## Code 
The folder containes the scripts to fine-tune the pre-trained model and a quick music genre prediction algorithm using the weights of model trained on custom dataset. 

Target variable: Genres in the [GTZAN dataset](http://marsyasweb.appspot.com/download/data_sets/):

- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock

### Requirements

The codebase uses Keras running over Theano to perform the experiments. 
- Have [pip](https://pip.pypa.io/en/stable/installing/) 
- Suggested install: [virtualenv](https://virtualenv.pypa.io/en/stable/)

Python packages necessary specified in *requirements.txt* 
For running the project for inference/training:
```
 # Create environment
 virtualenv myenv
 # Activate environment
 myenv\Scripts\activate.bat
 # Install dependencies
 pip install -r requirements.txt
 
```

### Example Code
The trained model can be downloaded from [HERE](https://drive.google.com/file/d/15uRJjFRMQde0pakmlL_N-WXmTkBiR4Bn/view?usp=sharing)
Populate the folder **music** with songs. Fill the example list (.txt file) with the song names. 
```
 python inference.py
 
```

## Results

### Senorita - Camila Cabello Shawn Mendes 

![fig_sea](https://github.com/laishawadhwa//fidures/results.png) 


### Verbose output for a given song
![Results](https://github.com/jsalbert/Music-Genre-Classification-with-Deep-Learning/blob/master/figs/output.png?raw=true)


