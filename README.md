
# NLP on Twitter

* Data used: [datahub.io/Twitter 2012 Presidential Election](https://datahub.io/dataset/twitter-2012-presidential-election)
* Search Engine: [OpenDataNetwork/Twitter](https://www.opendatanetwork.com/search?q=twitter&categories=)

## Relating tweets to hashtags using GLoVE embeddings and neural networks
Twitter JSON is processed to extract text, hashtags, and data structures relating the two. For each hashtag, [GLoVE embeddings](https://nlp.stanford.edu/projects/glove/) of tweeted words are concatenated to form tweet embeddings.  These tweet embeddings are then used as inputs for a fully-connected neural network to predict a likely hashtag embedding vector. This result is compared with approximating a transformation between tweet embeddings and hashtag embeddings using least squares.

## Usage
Executing `python3 python/main.py` will run through a custom test-case. The test-case used for basic testing was a modified version of the first part of the Twitter dataset: `cache-0-first100000.json`, produced by using `head -n 100000 cache-0.json > cache-0-first100000.json` in bash.

The custom test-case does JSON parsing, simple text cleaning, and builds data structures to make training multiple models simpler. A map between hashtags and associated tweets is used to calculate embeddings, a map between tweets and hashtags is used to train the neural network or construct matrices for least squares.

`tw_download.py` is a script to download Twitter JSON using [Tweepy](http://www.tweepy.org/). If being used outside of this project, the authentication keys should be replaced with new ones from Twitter.
