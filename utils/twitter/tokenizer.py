# Shishir Tandale

import re
class Tweet_Tokenizer(object):

    @staticmethod
    def build_character_map(tweets, maximum_size=None, batch_size=160):
        #returns character map, and encoded tweet string
        encoder = dict()
        decoder = dict()
        _tweets = tweets if maximum_size is None else tweets[:maximum_size]
        tweet_str = Tweet_Tokenizer.to_string(_tweets, batch_size)
        for char in tweet_str:
            if char not in encoder:
                encoder[char] = len(encoder.keys())
                decoder[encoder[char]] = char
        print(f"{len(encoder.keys())} unique characters")
        return (encoder, decoder), [float(encoder[c]) for c in tweet_str]

    @staticmethod
    def parse_character_map(encoded_tweets, character_map, batch_size=160):
        enc, dec = character_map
        tweet_str = [dec[int(t)] for t in encoded_tweets]
        tweets = "".join(([tweet_str[i:i+batch_size] for i in range(0, len(tweet_str), batch_size)])[0])
        return tweets

    @staticmethod
    def to_string(tweets, batch_size=160):
        padded_tweets = [str(t)+' '*max(batch_size-len(str(t)), 0) for t in tweets]
        return ''.join(padded_tweets)

    @staticmethod
    def apply_regex_to_text(text, replace_hashtags=True, replace_links=True, replace_user_refs=True):
        # remove hashtags, links, user_refs from tweets
        if replace_hashtags:
            #accept hashtags within words, optional trailing :
            r_hashtag = r"#\w+\b"
            text = re.sub(r_hashtag, "<hashtag>", text)
        if replace_links:
            #replace both twitter and pbs.twimg links not within words (malformed)
            #{4,} avoids selecting non twitter links with short base urls
            r_twlink = r"\bhttps?:\/\/(?:(?:twitter)|(?:pbs\.twimg)|(?:t))[./?=\w]{4,}\b"
            text = re.sub(r_twlink, "<url>", text)
        if replace_user_refs:
            #user refs can be 1 to 15 \w characters, not starting within words (avoid emails)
            r_usrrefs = r"\B@\w{1,15}:?\b"
            text = re.sub(r_usrrefs, "<user>", text)
        return text
