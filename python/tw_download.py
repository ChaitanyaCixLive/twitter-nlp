import sys
import tweepy
import json

def printJSONTweet(tweets):
	for tweet in tweets:
		print(json.dumps(tweet._json))

if len(sys.argv)<2:
	print("Expected two passed arguments: %s <twitterUsername> <numTweets<=3000>"%sys.argv[0])
	exit()

con_key = 'JLaq5gPoqprTBlD6f5L4n3EzD'
con_sec = '5XJtuLkFWigeAOcU9eDq14mK07S1PT3MCsNYZKqLAZq7IibghL'
acc_tok = '396595009-si5HIeXRODievxVqbRd9uYcwL0YKqaZoZwmxYagp'
acc_sec = 'TPt5PLAp8Y7UkiKJWKMDb9wkmfeMIJWphmX93TLfSiRJs'

auth = tweepy.OAuthHandler(con_key, con_sec)
auth.set_access_token(acc_tok, acc_sec)
api = tweepy.API(auth)

target_user = sys.argv[1] #get tweets from passed arg[1] 
max_tweets = int(sys.argv[2]) #get number of tweets from passed arg[2]
if max_tweets > 3000:
	print("You requested %d tweets, but the Twitter API max is 3000."%max_tweets);
	max_tweets = 3000

tweets = []

new_tweets = api.user_timeline(screen_name = target_user, count=200)
tweets.extend(new_tweets)
oldest = tweets[-1].id-1
printJSONTweet(new_tweets)

while len(tweets) < max_tweets: 
	new_tweets = api.user_timeline(screen_name = target_user, count=200, max_id=oldest)
	tweets.extend(new_tweets)
	oldest = tweets[-1].id-1
	printJSONTweet(new_tweets)
