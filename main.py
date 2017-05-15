# Shishir Tandale
import numpy as np

from utils.twitter_json import Tweet, Hashtag, User, Twitter_JSON_Parse
from utils.embeddings import Embedder
from models.baseline_model import Baseline_Model
from models.hybrid_vae_model import Hybrid_VAE_Model

def load_glove(glove_file, glove_shape):
    from collections import defaultdict
    lookup = defaultdict(lambda: -1, dict())
    glove = np.zeros(glove_shape)
    with open(glove_file, "r") as ef:
        for line_num, line in enumerate(ef):
            word, *embedding_vector = line.split(' ')
            lookup[word] = line_num
            glove[line_num] = [float(i) for i in embedding_vector]
    return glove, lookup

def main(num_tweets, num_hashtags, test_file, num_hashtags_print):
    #Load glove embeddings into Embedder
    glove_size = 1193514
    embedding_dim = 25
    glove_shape = (glove_size, embedding_dim)
    glove_file="../data/twitter-nlp/embeddings/glove.twitter.27B.25d.txt"
    print(f"Loading GloVe embeddings ({embedding_dim}).")
    glove_params = load_glove(glove_file, glove_shape)
    embedder = Embedder(glove_params)

    #load json from test_file, parse with Twitter_JSON_Parse
    json = open(test_file).readlines()[:num_tweets]
    twitter_parse = Twitter_JSON_Parse(json)
    tweets, hashtags = list(Tweet.set), list(Hashtag.set)
    print(f"Num tweets: {len(tweets)}, Num unique hashtags: {len(hashtags)}.")

    #package params and initialize baseline model
    print("Initializing Baseline Model.")

    blm = Baseline_Model(tweets, hashtags, embedding_dim)
    blm.create_embeddings(embedder)
    blm.train_model(epochs=5)
    sentences = [
        "Remarks at the United States Holocaust Memorial Museum's National Days of Remembrance.",
        "Today on Earth Day, we celebrate our beautiful forests, lakes and land. We stand committed to preserving the natural beauty of our nation.",
        "So sad to hear of the terrorist attack in Egypt. U.S. strongly condemns. I have great...",
        "Dems have been complaining for months & months about Dir. Comey. Now that he has been fired they PRETEND to be aggrieved. Phony hypocrites!"
    ]
    blm.predict(sentences, embedder)

    """
    #print("Encoding tweets for VAE.")
    #test encode some tweets
    character_count = 160
    batch_size = 20
    num_tweets = 20
    np_x = map_tweets_to_lookup(tweets[:num_tweets], glove_lookup, character_count)
    #x = tf.constant(np_x, dtype=tf.float32)
    print(f\"""Example tweet mapping:\n
            {tweets[0]}\t =>
            {tweets[0].text}\t =>\n
            {np_x[0]}
            \""")
    #build model and model graph
    #vae = Hybrid_VAE_Model()
    print("Building and training VAE.")
    #vae.build_graph(x)
    #vae.train()
    """

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--numtweets', type=int, default=100_000)
    parser.add_argument('-p', '--numprint', type=int, default=0)
    parser.add_argument('-a', '--hashtags', type=int, default=50)
    parser.add_argument('-t', '--testfile', default='../data/twitter-nlp/json/cache-0.json')
    args = parser.parse_args()
    main(
        args.numtweets,
        args.hashtags,
        args.testfile,
        args.numprint
    )
