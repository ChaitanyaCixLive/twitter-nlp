import sys
import json
import re

def containsHashtags(tweet):
	return len(tweet['entities']['hashtags']) > 0
def getHashtags(tweet):
	#assumes hashtags exist
	ht2txt = lambda h: "#"+h["text"]
	return " ".join(list(map(ht2txt, tweet['entities']['hashtags'])))

# if argument is provided, open as input JSON file
# else, accept input from stdin (allow piping in)
if len(sys.argv) > 1:
	line_gen = open(sys.argv[1])
else:
	line_gen = sys.stdin

emptystr = "<EMPTY>"

count = 0
tweets = []
hashtags = []
for line in line_gen:
	tweet = json.loads(line)
	# for more documentation, visit:
	# https://dev.twitter.com/overview/api/tweets

	# only saves tweets with hashtags
	if containsHashtags(tweet):
		r_hashtag = "([#].*)"
		r_twlink = "(http[:][/]{2}t[.]co.*)"
		# remove all hashtags and twitter links from text
		text = (re.sub("|".join([r_hashtag,r_twlink]), "", tweet["text"]))
		# print formatted hashtags on seperate line
		hashtags_i = (getHashtags(tweet))

		if hashtags_i == "":
			hashtags_i = emptystr
		if text == "":
			text = emptystr

		tweets.append(text)
		hashtags.append(hashtags_i)


with open("tweets.txt","w") as f:
	for tweet in tweets:
		f.write(tweet + "\n")

with open("hashtags.txt","w") as f:
	for hashtagset in hashtags:
		f.write(hashtagset+"\n")
