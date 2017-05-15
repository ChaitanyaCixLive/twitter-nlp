# Shishir Tandale

from utils.twitter.objects import Tweet, Hashtag

class Twitter_JSON_Parse(object):
    def __init__(self, json_txt,
                show_progress=False, replace_hashtags=True,
                replace_user_refs=False, replace_links=True):
        import json
        self.show_progress = show_progress
        self.replace_links = replace_links
        self.replace_hashtags = replace_hashtags
        self.replace_user_refs = replace_user_refs
        #parse text into json objects
        if self.show_progress:
            from utils.progress import Progress
            with Progress("Parsing text into JSON Object", len(json_txt)) as up:
                self.tweet_JSON_objs = [u(json.loads(line)) for line in json_txt]
        else:
            print("Parsing text into JSON Object.")
            self.tweet_JSON_objs = [json.loads(line) for line in json_txt]
        #extract text from tweets
        self.process_tweets()

    def process_tweets(self):
        from utils.twitter.tokenizer import Tweet_Tokenizer
        # for more documentation, visit:
        # https://dev.twitter.com/overview/api/tweets
        def extract_text(tweet_json):
            try:
                tweet_json["text"]
            except KeyError:
                return

            orig_text = tweet_json["text"].lower().strip()
            tweet_text = Tweet_Tokenizer.apply_regex_to_text(orig_text, self.replace_hashtags, self.replace_links, self.replace_user_refs)
            tweet_obj = Tweet(tweet_text, orig_text)
            #get, process, and package hashtags (this also stores and counts them)
            hashtags = [Hashtag(h["text"].lower()) for h in tweet_json["entities"]["hashtags"]]
            #connect hashtags to tweet and vice versa
            tweet_obj.register_hashtags(hashtags)
            [h.register_tweet(tweet_obj) for h in hashtags]

        if self.show_progress:
            with Progress("Extracting tweets", len(self.tweet_JSON_objs)) as up:
                [up(extract_text(tweet_obj)) for tweet_obj in self.tweet_JSON_objs[0]]
        else:
            print("Extracting tweets.")
            [extract_text(tweet_obj) for tweet_obj in self.tweet_JSON_objs]
