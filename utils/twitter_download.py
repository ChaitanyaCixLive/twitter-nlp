from argparse import ArgumentParser
from utils.progress import Progress
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
	def retrieve(self, user, num_tweets, *oldest):
		num_tweets = min(200, num_tweets)
		if oldest:
			self.tweets.extend(self.api.user_timeline(screen_name = user, count=num_tweets, max_id=oldest))
		else:
			self.tweets.extend(self.api.user_timeline(screen_name = user, count=num_tweets))
		if num_tweets > 200:
			# Using recursive retrieve to get around API limit
			self.retrieve(user, num_tweets-200, self.tweets[-1].id-1)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--user', default='UddiptoDutta')
	parser.add_argument('--count', type=int, default=1000)
	parser.add_argument('-p', '--print_tweets', action='store_true')
	args = parser.parse_args()
	client = TweepyClient()
	client.retrieve(args.user, args.count)
	if args.print_tweets:
		print(client.tweets)
