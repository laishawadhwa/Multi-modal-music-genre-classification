## CGM Course project
Faculty Instructor: Dr. Prerana Mukherjee 

Project Title: Music Genre classification using Deep Neural Networks (Group no. 4) 

Problem Statement: To develop a compact yet effective music genre classification system which can be used in recommendation systems (like Spotify and Gaana) [low computation at inference time] using multi modal feature fusion. 

## Motivation
Music Genre classification aims at predicting music genre using the audio signal. Being able to automate the task of detecting 
musical tags allows to create interesting content for the user like, music discovery and playlist creations, and for the 
content providers like music labeling and ordering. Determining music genres is the first step in that direction.  
Recent approaches use Deep Neural Networks (DNNs), that unify feature extraction and decision taking (classifiers). 
This allows learning the relevant features for each task at the same time that the system is learning to classify them.  
Several DNN-related algorithms have been proposed for automatic music tagging. Multi-resolution spectrograms are used to leverage the information in the audio signal on different 
time scales. in some papers pretrained weights of multilayer perceptrons are transferred in order to predict tags for other datasets. A two-layer convolutional network is used with mel-spectrograms as well as 
raw audio signals as input features. In a nutshell, the CNN and CRNN have been used where the  CNNs and RNNs play the roles of feature extractor and temporal summariser, respectively. 
Most experiments and results have been generated on GTZAN dataset which consists of 1000 music excerpts of 30 seconds duration 
with 100 examples in each 10 different music genres hence the methods utilise only one frame of around (29.12 s) per song.


## Novelties:

1.	A new text + music data-set with 300 songs and corresponding lyrics spanned across 10 genres is gebrated.
2.	Multi-frame strategy with an average stage developed.
3.	A multi modal fusion network approach for classification using both lyrics and mel spectograms.
4.	A Machine Learning and text ensemble for classification using time domain and frequency domain features. 

## Application use case:

- Music is the most popular art form that is performed and listened to by billions of people every day. There are many genres of music such as pop, classical, jazz, folk etc. Each genre has different music instruments, tone, rhythm, beats, flow etc. Digital music and 
online streaming has become very popular these days due to the increase in the number of users. 
- Many companies nowadays use music classification, either to recommend playlists to their customers (such as Spotify, Soundcloud) or simply as a product (Shazam, musixmatch). Music genre classification forms the basis steps for any music recommendation system.

For each of the network three separate folders have been created.
1.  Multi-frame strategy with an average stage developed: [Multiframe Approach](https://github.com/laishawadhwa/CGM_Project/tree/master/Group_4/Multiframe%20Approach)
2.	A multi modal fusion network approach for classification using both lyrics and mel spectograms: [Dense Co Attention](https://github.com/laishawadhwa/CGM_Project/tree/master/Group_4/Dense%20Co%20Attention)
3.	A Machine Learning and text ensemble for classification using time domain and frequency domain features: [XGBOOST](https://github.com/laishawadhwa/CGM_Project/tree/master/Group_4/XGBOOST)


Medium Post can be found [HERE](https://medium.com/@laisha.w16_85978/understanding-music-genre-classification-a-multi-modal-fusion-approach-6989caa87803) 

The running instructions are provided in each of the folders.

## Following are the results

![Results for experiments](https://github.com/laishawadhwa/CGM_Project/blob/master/Group_4/Tableres.PNG)
