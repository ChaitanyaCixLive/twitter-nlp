import tweepy, json

class Tweepy_Client(object):
	def __init__(self,
			con_key="JLaq5gPoqprTBlD6f5L4n3EzD",
			con_sec="5XJtuLkFWigeAOcU9eDq14mK07S1PT3MCsNYZKqLAZq7IibghL",
			acc_tok="396595009-si5HIeXRODievxVqbRd9uYcwL0YKqaZoZwmxYagp",
			acc_sec="TPt5PLAp8Y7UkiKJWKMDb9wkmfeMIJWphmX93TLfSiRJs"
			):
		self.con_key = con_key
		self.con_sec = con_sec
		self.acc_tok = acc_tok
		self.acc_sec = acc_sec
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

	def start_stream(self, filter_text, listener):
		self.stream = tweepy.Stream(auth=self.auth, listener=listener)
		try:
			self.stream.filter(languages=["en"], track=["#", filter_text])
		except KeyboardInterrupt:
			print("Stream Interrupted")

class Tweepy_Stream_Printer(tweepy.StreamListener):
	def on_status(self, status):
		print(status.text)
	def on_error(self, status_code):
		if status_code == 420:
			return False

class Tweepy_Stream_Saver(tweepy.StreamListener):
	buffer = None
	def __init__(self, max_count=10_000):
		if Tweepy_Stream_Saver.buffer is None:
			Tweepy_Stream_Saver.buffer = []
		self.max_count = max_count
		self.i = 0
	def on_data(self, data):
		Tweepy_Stream_Saver.buffer.append(data)
		self.i += 1
		if self.i >= self.max_count:
			return False

	def on_error(self, status_code):
		if status_code == 420:
			return False

if __name__ == "__main__":
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--user', default='sundarpichai')
	parser.add_argument('--count', type=int, default=1000)
	parser.add_argument('--filter', default='lang:en')
	parser.add_argument('-p', '--print_tweets', action='store_true')
	args = parser.parse_args()
	client = Tweepy_Client()
	tss = Tweepy_Stream_Saver()
	client.start_stream(args.filter, tss)
	print(tss.buffer)
