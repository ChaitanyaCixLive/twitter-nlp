
# Tweet Generation and Hashtag Prediction

#### Data used:
- [datahub.io/Twitter 2012 Presidential Election][1]
- [OpenDataNetwork/Twitter](https://www.opendatanetwork.com/search?q=twitter&categories=)

#### Papers referenced:
- [A Hybrid Convolutional Variational Autoencoder for Text Generation][2]

## Generating new tweets
A *hybrid convolutional variational autoencoder*, inspired by [Semeniuta, Severyn, and Barth's
2017 paper][2], is used to encode tweet character vectors into a low-dimensional hidden
representation. This hidden representation is then used as the basis for generation of new tweets.

## Predicting hashtags from tweets
Twitter JSON is processed to extract text, hashtags, and data structures relating the two.

For each hashtag, [GLoVE embeddings](https://nlp.stanford.edu/projects/glove/) are averaged
together to form tweet embeddings and then hashtag embeddings from averaging tweets together.
These tweet embeddings are then used as inputs for a fully-connected neural network. The network
response is fed through a softmax classifier and euclidean distance (l2) is used as the loss
function. For hashtag prediction, sentence embeddings are calculated in the same way by
averaging word vectors together, then they are fed through the neural network and the predicted
vectors are searched in our calculated hashtag embeddings using nearest neighbor.

## Usage
This project targets python 3.6.0, tensorflow 1.0.1

Executing `python main.py --help` shows the available arguments. By default, 100,000 tweets
are extracted from the JSON file, zero hashtags are printed out, and 50 are embedded. The default
test file is assumed to be `cache-0.json` from the [Twitter 2012 election][1] dataset located at
`../data/twitter-nlp/json/cache-0.json`. Other parameters, such as the GloVe embedding dimension
for the baseline hashtag prediction model and the embedding file are specified within `main.py`.
Currently, tweet and hashtag embeddings are created with the Embedder and then trained with the
Baseline Model. Afterwards, tweets are loaded into the hybrid VAE model.

## Included Utilities
`utils.twitter_download` contains a client for [Tweepy](http://www.tweepy.org/). This allows
custom Twitter datasets to be downloaded given a username and a count. There is an API limit of 200
tweets per request, but this tool provides a recursive solution around this that maxes out
the assigned Twitter API account's download limit.

If being used outside of this project, the authentication keys should be replaced with new ones from Twitter.
#### Example
```python
from utils.twitter_download import Tweepy_Client
#default args (do not use outside of this project!)
client = Tweepy_Client()
#custom authentication keys
client = Tweepy_Client(con_key=" ",
                      con_sec=" ",
                      acc_tok=" ",
                      acc_sec=" ")
client.retrieve("sundarpichai", 500)
#raw json is contained in client.tweets
print(client.tweets)
```
[1]: https://datahub.io/dataset/twitter-2012-presidential-election "Twitter 2012 Presidential Election"
[2]: https://arxiv.org/pdf/1702.02390.pdf "A Hybrid Convolutional Variational Autoencoder for Text Generation"
