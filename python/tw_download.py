import tweepy, json

class TweepyClient(object):
	def __init__(self):
		self.con_key = 'JLaq5gPoqprTBlD6f5L4n3EzD'
		self.con_sec = '5XJtuLkFWigeAOcU9eDq14mK07S1PT3MCsNYZKqLAZq7IibghL'
		self.acc_tok = '396595009-si5HIeXRODievxVqbRd9uYcwL0YKqaZoZwmxYagp'
		self.acc_sec = 'TPt5PLAp8Y7UkiKJWKMDb9wkmfeMIJWphmX93TLfSiRJs'
		self.auth = tweepy.OAuthHandler(self.con_key, self.con_sec)
		self.auth.set_access_token(self.acc_tok, self.acc_sec)
		self.api = tweepy.API(self.auth)
		self.tweets = []
	def retrieve(self, user, numTweets, *oldest):
		numTweetsToGet = min(200, numTweets)
		if oldest:
			self.tweets.extend(self.api.user_timeline(screen_name = user, count=numTweetsToGet, max_id=oldest))
		else:
			self.tweets.extend(self.api.user_timeline(screen_name = user, count=numTweetsToGet))
		if numTweets > 200:
			# Using recursive retrieve to get around API limit
			self.retrieve(user, numTweets-200, self.tweets[-1].id-1)

def printJSONTweets(tweets):
	parsedTweets = [json.dumps(tweet._json) for tweet in tweets]
	print(parsedTweets)
def main(twitterUsername, count):
	tweepy = TweepyClient()
	tweepy.retrieve(twitterUsername, count)
	printJSONTweets(tweepy.tweets)

if __name__ == "__main__":
	main("shishirtandale", 1)
