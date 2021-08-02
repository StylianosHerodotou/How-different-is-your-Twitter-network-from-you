import tweepy
import json
import pandas as pd
import csv
import re #regular expression
import numpy as np
import pymongo
from pymongo import MongoClient
from pymongo.errors import WriteError
from textblob import TextBlob
import string
import preprocessor as p
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random



nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet

data = pd.read_csv("/home/ubuntu/Desktop/DATASET_Crunchbase_Founders_with_Twitter v2.csv", delimiter="\t")
list_of_people = data["twitter_username"]
consumer_key = ""
consumer_secret_key = ""
access_tokken = ""
access_tokken_secret = ""
# pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_tokken, access_tokken_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True,
                 parser=tweepy.parsers.JSONParser())


def user_already_in_collection(screen_name,collection)->bool:
    if(collection.find_one({"screen_name":screen_name.lower() })==None):
        # print(screen_name, " not in database yet")
        return False
    else:
        # print(screen_name, "already in databse")
        return True


def save_users_in_collection(users:list,collection,
    first_index:int =0,last_index=-1):
    for i in range (len(users)-first_index):
        index=i+first_index
        if(index==last_index):
            break;
        user=users[index]
        save_user_in_collection(user,collection)
        if(i%100==0):
            print("i have saved ",(index+1), " users")

def save_user_in_collection(screen_name:str, collection):
    if(user_already_in_collection(screen_name,collection)==False):
        try:
            user = api.get_user(screen_name)
            user=dict((user))
            if '_api' in user:
                del user['_api']
            if 'status' in user:
                del user['status']
            user["_id"] = user.pop("id")
            low=user["screen_name"].lower()
            user["screen_name"]=low
            ans=collection.insert_one(user)
            #print(screen_name ," is insterted")
            return ans
        except tweepy.error.TweepError as e:
            print("there is no user with screen name ", screen_name)
            return None
        except pymongo.errors.DuplicateKeyError as e:
            print("attempted dublicate key. with screen name",screen_name)
            return None
    else:
        #print(screen_name ," was already in database")
        return None

def turn_screen_name_to_lowercase(collection):
    for user in collection.find():
        prin=user["screen_name"]
        meta=prin.lower()
        # if(meta!=prin):
        #     print("prin ",prin, " meta ", meta)
        collection.update({"_id": user["_id"]}, {"$set": {"screen_name":  meta}})



def get_all_followers_of_user(screen_name:str, poses_selides_na_psaksi:int=1)->list:
    followers_ids = []
    try:
        for page in tweepy.Cursor(api.followers_ids, screen_name=screen_name).pages(poses_selides_na_psaksi):
            followers_ids.extend(page["ids"])
        return followers_ids
    except tweepy.error.TweepError:
        return followers_ids


def add_followers_to_user(user: dict, collection):
    if ("followers_ids" in user.keys()):
        print("user ", user["screen_name"], " already had followers_ids")
        return None
    screen_name = user["screen_name"]
    followers_ids = get_all_followers_of_user(screen_name)
    try:
        collection.update_one({"_id": user['_id']}, {"$set": {"followers_ids": followers_ids}})
    except:
        return None
    return user


def add_followers_to_users(everything: list, collection):
    new_everything = []
    for i, user in enumerate(everything):
        new = add_followers_to_user(user, collection)
        if (i % 10 == 0):
            print("we are at ", i)
    return

def get_all_friends_of_user(screen_name:str, poses_selides_na_psaksi:int=1)->list:
    friends_ids = []
    try:
        for page in tweepy.Cursor(api.friends_ids, screen_name=screen_name).pages(poses_selides_na_psaksi):
            friends_ids.extend(page["ids"])
        return friends_ids
    except tweepy.error.TweepError:
        return friends_ids


def add_following_to_user(user: dict, collection):
    if ("following_ids" in user.keys()):
        print("user ", user["screen_name"], " already had following_ids")
        return None
    screen_name = user["screen_name"]
    following_ids = get_all_friends_of_user(screen_name)
    try:
        collection.update_one({"_id": user['_id']}, {"$set": {"following_ids": following_ids}})
    except:
        return None
    return user


def add_following_to_users(everything: list, collection):
    new_everything = []
    for i, user in enumerate(everything):
        new = add_following_to_user(user, collection)
        if (i % 100 == 0):
            print("we are at ", i)
    return


def generate_n_uniqe_random_integers(starting:int=0,final:int =-1,
                                n=10):
    ans=random.sample(range(starting, final), n)
    return ans

def find_and_add_all_tweets_of_user(user_id: int, unprocessed, processed,max_tweets_per_user=3240):
    # Twitter only allows access to a users most recent 3240 tweets with this method
    all_tweet_count = 0

    # initialize a list to hold all the tweepy Tweets and the tweet data, soon to be the df
    tweets_data = []
    all_tweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    try:
        new_tweets = api.user_timeline(user_id=user_id, count=200)
    except:
        print("unable to get tweets for user id ",user_id)
        return tweets_data
    all_tweets = all_tweets + new_tweets
    all_tweet_count = all_tweet_count + len(new_tweets)
    # process the first tweets and add them to the tweets_data
    processed_tweets = process_these_tweets(new_tweets)
    tweets_data = tweets_data + processed_tweets
    for tweet in new_tweets:
        tweet["_id"]=tweet.pop("id")
    try:
        if (len(new_tweets) > 0):
            unprocessed.insert_many(new_tweets)
        if (len(processed_tweets) > 0):
            processed.insert_many(processed_tweets)
    except:
        print("i was unable to save evrything to the database")
        continue

    # save the id of the oldest tweet less one
    if(len(new_tweets)>0):
        oldest = new_tweets[-1]["_id"] - 1
    else:
        oldest=None

    i = 0  # tuto prepi na fii meta to testing
    # keep grabbing tweets until there are no tweets left to grab
    while (len(new_tweets) and (i < 1) and all_tweet_count<=max_tweets_per_user) > 0:
        print(f"getting tweets before {oldest}")

        # all subsiquent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(user_id=user_id, count=200, max_id=oldest)
        except:
            print("unable to get the rest of the tweets for user id  ", user_id)
            return tweets_data
        all_tweets = all_tweets + new_tweets
        all_tweet_count = all_tweet_count + len(new_tweets)
        # process the tweets and add them to the tweets_data
        processed_tweets = process_these_tweets(new_tweets, column_keys)
        tweets_data = tweets_data + processed_tweets
        for tweet in new_tweets:
            tweet["_id"] = tweet.pop("id")
        if(len(new_tweets)>0):
            oldest = new_tweets[-1]["_id"] - 1
        else:
            oldest=None
        try:
            if (len(new_tweets) > 0):
                unprocessed.insert_many(new_tweets)
                # update the id of the oldest tweet less one
            if (len(processed_tweets) > 0):
                processed.insert_many(processed_tweets)
        except:
            print("i was unable to save evrything to the database")
            continue

        print(f"...{all_tweet_count} tweets downloaded so far")
        i = i + 1  # tuto ena prepi na fii meta to testing.

    return tweets_data


# Sad Emoticons
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                 '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}
#HappyEmoticons
emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}
#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)

def find_a_count(text):
    lista=text.split()
    count=0
    for word in lista:
        if (word.startswith("@") and (len(word)>1)):
            count=count+1
    return count

def removeContractions(text:str)->str:
    w_tokenizer = TweetTokenizer()
    lista= []
    keep=True
    for w in w_tokenizer.tokenize((text)):
        if(w=="?"):
            keep=False
        elif(keep==False):
            keep= True
        else:
            lista.append(w)
  #  print(len(lista))
    ans=' '.join(lista)
    return ans

def remove_punctuation(words):
    return ' '.join(word.strip(string.punctuation) for word in words.split())

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = TweetTokenizer()
    lista= []
    for w in w_tokenizer.tokenize((text)):
         lista.append(lemmatizer.lemmatize(w,get_wordnet_pos(w)))
    ans=' '.join(lista)
    return ans

def extract_hash_tags(s):
    return set(part[1:] for part in s.split() if part.startswith('#'))


def clean_description(dirty: str) -> str:
    stop_words = set(stopwords.words('english'))

    # gia na eshi ta new lines
    tweet = dirty.replace('\n', ' ').replace('\r', '')
    # Convert text to lowercase
    tweet = tweet.lower()
    #     print("lower",tweet)
    # removes punctuation
    tweet = remove_punctuation(tweet)
    #     print("punctuation",tweet)
    # remove contractions
    tweet = removeContractions(tweet)
    #     print("contractions",tweet)
    # clean this shit
    tweet = p.clean(tweet)
    #     print("cleaned it",tweet)
    # after tweepy preprocessing the colon symbol left remain after removing mentions

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    #     print("ascii char",tweet)
    tweet = re.sub(r':', '', tweet)
    #     print(tweet)
    tweet = re.sub(r'Ä¶', '', tweet)
    #     print(tweet)
    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    # print("emojies",tweet)
    # remove numbers
    tweet = re.sub(r"\d+", "", tweet)
    # print("remove numbers",tweet)

    # lemmatize text
    tweet = lemmatize_text(tweet)
    # print("lem",tweet)

    # token word
    word_tokens = word_tokenize(tweet)
    # remove stopwords
    clean = []
    w_t = []
    for i in word_tokens:
        if i not in stop_words:
            w_t.append(i)
    word_tokens = w_t

    #  print("stop words",' '.join(word_tokens))
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in emoticons:
            clean.append(w)
    return ' '.join(clean)

def process_description(description:str)->str:

    original_text = description
    clean_text = clean_description(original_text)
    # find polarity and subjectiviy
    blob = TextBlob(clean_text)
    Sentiment = blob.sentiment
    polarity = Sentiment.polarity
    subjectivity = Sentiment.subjectivity
    # find hashtags
    hashtag_count = len(extract_hash_tags(original_text))
    # find user_mentions
    user_mention_count = find_a_count(original_text)
    # find sentiment
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(original_text)
    sentiment = vs["compound"]  # mono tuto valo ala maybe ena theli je ta alla 3. tra na dume.
    positive = vs["pos"]
    negative = vs["neg"]
    neutral = vs["neu"]

    # neg=negative, neu=neutral , pos=positive compound vasika
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    # -1 (most extreme negative) and +1 (most extreme positive).
    # me to for vali ta columns p en mesti lista colums.
    discreption_dict = {
                         "polarity": polarity,
                         "subjectivity": subjectivity,
                         "hashtag_count": hashtag_count,
                         "user_mention_count": user_mention_count,
                         "sentiment": sentiment
                         }

    return discreption_dict


def clean_tweet(dirty: str) -> str:
    stop_words = set(stopwords.words('english'))

    # gia na eshi ta new lines
    tweet = dirty.replace('\n', ' ').replace('\r', '')
    # Convert text to lowercase
    tweet = tweet.lower()
    #     print("lower",tweet)
    # removes punctuation
    tweet = remove_punctuation(tweet)
    #     print("punctuation",tweet)
    # remove contractions
    tweet = removeContractions(tweet)
    #     print("contractions",tweet)
    # clean this shit
    tweet = p.clean(tweet)
    #     print("cleaned it",tweet)
    # after tweepy preprocessing the colon symbol left remain after removing mentions

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    #     print("ascii char",tweet)
    tweet = re.sub(r':', '', tweet)
    #     print(tweet)
    tweet = re.sub(r'Ä¶', '', tweet)
    #     print(tweet)
    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    # print("emojies",tweet)
    # remove numbers
    tweet = re.sub(r"\d+", "", tweet)
    # print("remove numbers",tweet)

    # lemmatize text
    tweet = lemmatize_text(tweet)
    # print("lem",tweet)

    # token word
    word_tokens = word_tokenize(tweet)
    # remove stopwords
    clean = []
    w_t = []
    for i in word_tokens:
        if i not in stop_words:
            w_t.append(i)
    word_tokens = w_t

    #  print("stop words",' '.join(word_tokens))
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in emoticons:
            clean.append(w)
    return ' '.join(clean)


def process_these_tweets(tweets):
    processed_tweets = []
    for tweet in tweets:
        processed_tweet = process_these_tweet(tweet)
        processed_tweets.append(processed_tweet)
    return processed_tweets


def process_these_tweet(tweet):
    status = tweet
    status_dict = dict(status)
    # ta keys p en eshi apefthias pio kato.
    user_id = status_dict["user"]["id"]
    user = status_dict["user"]["screen_name"]
    original_text = status_dict["text"]
    clean_text = clean_tweet(original_text)
    # find polarity and subjectiviy
    blob = TextBlob(clean_text)
    Sentiment = blob.sentiment
    polarity = Sentiment.polarity
    subjectivity = Sentiment.subjectivity
    # find hashtags
    hashtags = []
    lod_for_hashtags = status_dict["entities"]["hashtags"]
    for dic in lod_for_hashtags:
        hashtags.append(dic["text"])
    # find user_mentions
    user_mentions = []
    lod_for_user_mentions = status_dict["entities"]["user_mentions"]
    for dic in lod_for_user_mentions:
        user_mentions.append(dic["screen_name"])
    # find sentiment
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(original_text)
    sentiment = vs["compound"]  # mono tuto valo ala maybe ena theli je ta alla 3. tra na dume.
    positive = vs["pos"]
    negative = vs["neg"]
    neutral = vs["neu"]
    # find id
    inti = status_dict["_id"]
    is_quote = status_dict["is_quote_status"]
    retweet_count = status_dict["retweet_count"]
    favorite_count = status_dict["favorite_count"]
    is_reply = None
    if (status_dict["in_reply_to_screen_name"] != None):
        is_reply = True
    else:
        is_reply = False

    # neg=negative, neu=neutral , pos=positive compound vasika
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    # -1 (most extreme negative) and +1 (most extreme positive).
    # me to for vali ta columns p en mesti lista colums.
    ans = {"_id": inti,
           "user": user,
           "user_id": user_id,
           "original_text": original_text,
           "clean_text": clean_text,
           "polarity": polarity,
           "subjectivity": subjectivity,
           "hashtag_count": len(hashtags),
           "user_mention_count": len(user_mentions),
           "sentiment": sentiment,
           "is_quote": is_quote,
           "retweet_count": retweet_count,
           "favorite_count": favorite_count,
           "is_reply": is_reply

           }
    return ans

def process_and_add_tweets_to_database(unpne_tweets_col,processed_tweets_collection,tweets_os_tora=0):
    flag=False
    count_of_tweets=processed_tweets_collection.count()
    while(flag!=True):
        unpne_tweets=unpne_tweets_col.find({}).skip(count_of_tweets).limit(200)
        if(unpne_tweets==None):
            flag=True
            break
        try:
            unpne_tweets=list(unpne_tweets)
        except:
            continue
        count_of_tweets=count_of_tweets+len(unpne_tweets)
        processed_tweets=process_these_tweets(unpne_tweets)
        try:
            processed_tweets_collection.insert_many(processed_tweets)
        except:
            print("i could not add all of them")

def find_and_add_all_tweets_of_users(all_users,unprocessed,processed):
    for user in all_users:
        find_and_add_all_tweets_of_user(user["_id"],unprocessed,processed)


def get_all_followers_users(entrepreneurs, follower_collection):
    for entrepreneur in entrepreneurs:
        followers_ids = entrepreneur["followers_ids"]
        for follower_id in followers_ids:
            # check an en idi mesa
            if (follower_collection.find_one({"_id": follower_id}) == None):
                follower = api.get_user(user_id=follower_id)
                follower = dict(follower)
                follower["_id"] = follower.pop("id")
                follower["follows"] = entrepreneur["_id"]
                follower["entrepreneur"] = False
                try:
                    follower_collection.insert_one(follower)
                except WriteError as e:
                    print("could not add this dude ", follower_id)
            else:
                continue


def get_all_following_users(entrepreneurs, following_collection):
    for entrepreneur in entrepreneurs:
        following_ids = entrepreneur["following_ids"]
        for following_id in following_ids:
            # check an en idi mesa
            if (following_collection.find_one({"_id": following_id}) == None):
                following = api.get_user(user_id=following_id)
                following = dict(following)
                following["_id"] = following.pop("id")
                following["follows"] = entrepreneur["_id"]
                following["entrepreneur"] = False
                try:
                    following_collection.insert_one(following)
                except WriteError as e:
                    print("could not add this dude ", following_id)
            else:
                continue

def get_all_tweets_from_not_entrepreneurs(entrepreneurs,unprocessed,processed,final_sample_collection,
        max_tweets_per_user=3240,max_friends_per_user=5000):

    misoi=math.ceil(len(entrepreneurs) / 2)
    for i,entrepreneur in enumerate(entrepreneurs):

        if(i>=misoi):
            break;
        if("tweets_collected_from_followers_and_followings" in entrepreneur.keys() and entrepreneur["tweets_collected_from_followers_and_followings"]==True):
            print("already collected followers and followings from ",entrepreneur["screen_name"])
            continue
        else:
            #generate random sample
            random_sample_indexes=generate_n_uniqe_random_integers(starting=0,
                                            final= misoi, n = max_friends_per_user):
            random_sample=[]
            for(index in random_sample_indexes):
                random_sample.append(entrepreneur[index])

            for follower_id in random_sample["followers_ids"]:
                find_and_add_all_tweets_of_user(follower_id,unprocessed,processed,max_tweets_per_user)
            for following_id in random_sample["following_ids"]:
                find_and_add_all_tweets_of_user(following_id,unprocessed,processed,max_tweets_per_user)
            try:
                final_sample_collection.update_one({"_id":entrepreneur['_id']},{"$set" : {"tweets_collected_from_followers_and_followings":True}})
            except:
                print("unable na vali to pedio tweets_collected_from_followers_and_followings sto xristi ",entrepreneur["screen_name"])
                continue

def get_everything_to_a_list(collection, start_from_beggining=False,per_time=200):
    flag = False
    everything=[]
    if(start_from_beggining==True):
        count=0
    else:
        count = collection.count()
    while (flag != True):
        new_staff = collection.find({}).skip(count).limit(per_time)
        if (new_staff == None):
            flag = True
            break
        try:
            new_staff = list(new_staff)
        except:
            continue
        count = count + len(new_staff)
        everything = everything+new_staff
    return everything



def main():
    client = MongoClient('localhost', 27017)
    db = client["testDB"]
    processed = db["processed_tweets_friends_and_followers"]
    unprocessed = db["unprocessed_tweets_friends_and_followers"]
    final_sample_collection = db["final_sample"]

    entrepreneurs = get_everything_to_a_list(final_sample_collection)
    max_tweets_per_user = 600
    max_friends_per_user = 150
    get_all_tweets_from_not_entrepreneurs(entrepreneurs, unprocessed, processed,
        final_sample_collection,max_tweets_per_user,max_friends_per_user)
    # get_all_followers_users(final_sample, followers_collection)

    #turn_screen_name_to_lowercase(collection)
if __name__ == "__main__":
    main()