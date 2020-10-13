# Co Attention network: Music genre classification

Most genre classification techniques for music either use the musical features or the lyrics to classify the song however using them together boosts the accuracy and results in a lesser false positives across the genres
The novel Dense Co-Attention Network with KQV Attention and Multi-Modal Feature Fusion technique has been used for fusing the text and image embeddings together into one embedding and classifying the genre.

The input to the network is the Image (mel spectogram of the song frame) and text (lyrics of the song).

The Network usese KQV attention to find the features of lyrics important for classification w.r.t the mel spectogram.

The network trained on a collection of 50000 songs and corresponding lyrics gives an overall accuracy of 90.4%.

The repository provides the training code for the co attention network.

Input to networK Image + text

output : label (genre)

