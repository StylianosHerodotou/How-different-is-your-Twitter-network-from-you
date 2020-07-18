import tweepy
import json
import numpy as np
import pandas as pd
import csv
import re #regular expression


import datetime
from datetime import date

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
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
import math
import random
import multiprocessing

import pymongo
from pymongo import MongoClient

# data = pd.read_csv("/home/ubuntu/Desktop/DATASET_Crunchbase_Founders_with_Twitter v2.csv", delimiter="\t")
# list_of_people = data["twitter_username"]
#first

# # pass twitter credentials to tweepy
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
# auth.set_access_token(access_tokken, access_tokken_secret)
# api = tweepy.API(auth, wait_on_rate_limit=True,
#                  wait_on_rate_limit_notify=True,
#                  parser=tweepy.parsers.JSONParser())


def generate_random_sample(full_list, starting, final, sample_size):
    # generate random sample
    if(final>len(full_list)):
        final=len(full_list)
    if(sample_size>len(full_list)):
        s=len(full_list)
    else:
        s=sample_size
    random_sample_indexes = generate_n_uniqe_random_integers(starting=starting, final=final, n=s)
    random_sample = []
    for index in random_sample_indexes:
        random_sample.append(full_list[index])
    return random_sample


def get_everything_to_a_list(collection, start_from_beggining=False,per_time=200):
    flag = False
    everything=[]
    count = collection.estimated_document_count()
    if(count==0):
        return []
    if(start_from_beggining==True):
        count=0
    print(count)

    while (flag != True):
        new_staff = collection.find({}).skip(count).limit(per_time)
        if (new_staff == None):
            flag = True
            break
        try:
            new_staff = list(new_staff)
            print(new_staff)
            if(len(new_staff)==0):
                flag=True
                break
        except:
            continue
        count = count + len(new_staff)
        everything = everything+new_staff
    return everything


def generate_n_uniqe_random_integers(starting: int = 0, final: int = -1,
                                     n=10):
    ans = random.sample(range(starting, final), n)
    return ans

#

def process_these_tweets(tweets):
    processed_tweets = []
    for tweet in tweets:
        processed_tweet = process_this_tweet(tweet)
        processed_tweets.append(processed_tweet)
    return processed_tweets


def process_this_tweet(tweet):
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
    inti = status_dict["id"]
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


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def get_current_date():
    temp = datetime.today().strftime('%d-%m-%Y')
    temp = temp.split("-")
    ans_date = (int(temp[0]), int(temp[1]), int(temp[2]))
    return ans_date


def get_days_so_far(today, then):
    d_today = date(today[2], today[1], today[0])
    d_then = date(then[2], then[1], then[0])
    days = d_today - d_then
    return days.days


def find_number_of_posts_per_week(number_of_posts, created_at):
    then = find_date(created_at)
    today = get_current_date()
    days_between = get_days_so_far(today, then)
    ans = number_of_posts / days_between
    return ans


import re

# Sad Emoticons
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                 '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}
# HappyEmoticons
emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}
# Emoji patterns
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
    lista = text.split()
    count = 0
    for word in lista:
        if (word.startswith("@") and (len(word) > 1)):
            count = count + 1
    return count


def removeContractions(text: str) -> str:
    w_tokenizer = TweetTokenizer()
    lista = []
    keep = True
    for w in w_tokenizer.tokenize((text)):
        if (w == "?"):
            keep = False
        elif (keep == False):
            keep = True
        else:
            lista.append(w)
    #  print(len(lista))
    ans = ' '.join(lista)
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
    lista = []
    for w in w_tokenizer.tokenize((text)):
        lista.append(lemmatizer.lemmatize(w, get_wordnet_pos(w)))
    ans = ' '.join(lista)
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


def process_description(description: str) -> str:
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


def get_avg_sentiment_of_user(user_id,collection):
    all_tweeets_of_user = collection.find({"user": {"id": user_id}})
    count = 0
    for i, tweet in enumerate(all_tweeets_of_user):
        sum_sentiment = sum_sentiment + tweet["sentiment"]
        count = i + 1

    return sum_sentiment / count


def find_followers_following_ratio(follower_count, following_count):
    return follower_count / following_count


def find_date(date_str: str):
    months = {'Jan': 1,
              "Feb": 2,
              "Mar": 3,
              "Apr": 4,
              "May": 5,
              "Jun": 6,
              "Jul": 7,
              "Aug": 8,
              "Sep": 9,
              "Oct": 10,
              "Nov": 11,
              "Dec": 12}

    splitted = date_str.split()
    month = months[splitted[1]]
    day = splitted[2]
    year = splitted[-1]
    try:
        ans_date = (int(day), month, int(year))
    except:
        ans_date = None
    return ans_date


def find_number_of_posts_per_week(number_of_posts, created_at):
    then = find_date(created_at)
    today = get_current_date()
    days_between = get_days_so_far(today, then)
    ans = number_of_posts / days_between
    return ans


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


# http://10.16.3.58:6670/?token=acf3308900a42c8fcd381091a486ff52ac7bc7539f3e2bdf


def find_and_add_tweets_of_users(sample_of_friends,unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,
                                     keep_track_of_tweets_collection,max_tweets_per_user,code_for_api):
    api=None
    if(code_for_api==1):
        auth = tweepy.OAuthHandler(consumer_key1, consumer_secret_key1)
        auth.set_access_token(access_tokken1, access_tokken_secret1)
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True,
                         parser=tweepy.parsers.JSONParser())
    else:
        auth = tweepy.OAuthHandler(consumer_key2, consumer_secret_key2)
        auth.set_access_token(access_tokken2, access_tokken_secret2)
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True,
                         parser=tweepy.parsers.JSONParser())
    for friend_id in sample_of_friends:
        find_and_add_tweets_of_user(user_id=friend_id, unprocessed=unprocessed_tweets_of_friends_collection, processed=processed_tweets_of_friends_collection,
                                        keep_track_of_tweets_collection=keep_track_of_tweets_collection,api=api,max_tweets_per_user=max_tweets_per_user)


def find_and_add_tweets_of_user(user_id: int, unprocessed, processed, keep_track_of_tweets_collection,api,
                                    max_tweets_per_user=3240):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    temp = keep_track_of_tweets_collection.find_one({"_id": user_id})

    if (temp == None):
        all_tweet_count = 0
    else:
        all_tweet_count = temp["count"]

    # initialize a list to hold all the tweepy Tweets and the tweet data, soon to be the df
    tweets_data = []
    all_tweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    try:
        if (temp == None or temp["oldest"] == -1):
            new_tweets = api.user_timeline(user_id=user_id, count=200)
        else:
            new_tweets = api.user_timeline(user_id=user_id, count=200, max_id=temp["oldest"])
    except:
        print("unable to get tweets for user id ", user_id)
        return tweets_data
    all_tweets = all_tweets + new_tweets
    all_tweet_count = all_tweet_count + len(new_tweets)
    # process the first tweets and add them to the tweets_data
    processed_tweets = process_these_tweets(new_tweets)
    tweets_data = tweets_data + processed_tweets
    for tweet in new_tweets:
        tweet["_id"] = tweet.pop("id")

    oldest = -1

    # save the id of the oldest tweet less one
    if (len(new_tweets) > 0):
        oldest = new_tweets[-1]["_id"] - 1

    try:
        if (len(new_tweets) > 0):
            unprocessed.insert_many(new_tweets)
        if (len(processed_tweets) > 0):
            processed.insert_many(processed_tweets)
    except:
        print("i was unable to save evrything to the database")

    # keep grabbing tweets until there are no tweets left to grab
    while ((len(new_tweets) > 0) and all_tweet_count < max_tweets_per_user) > 0:
        print(f"getting tweets before {oldest}")
        # all subsiquent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(user_id=user_id, count=200, max_id=oldest)
        except:
            print("unable to get the rest of the tweets for user id  ", user_id)
            break
        all_tweets = all_tweets + new_tweets
        all_tweet_count = all_tweet_count + len(new_tweets)
        # process the tweets and add them to the tweets_data
        processed_tweets = process_these_tweets(new_tweets)
        tweets_data = tweets_data + processed_tweets
        for tweet in new_tweets:
            tweet["_id"] = tweet.pop("id")
        if (len(new_tweets) > 0):
            oldest = new_tweets[-1]["_id"] - 1
        try:
            if (len(new_tweets) > 0):
                unprocessed.insert_many(new_tweets)
                # update the id of the oldest tweet less one
            if (len(processed_tweets) > 0):
                processed.insert_many(processed_tweets)
        except:
            print("i was unable to save evrything to the database")

        print(f"...{all_tweet_count} tweets downloaded so far")
    if (temp == None):
        q = {
            "_id": user_id,
            "count": all_tweet_count,
            "oldest": oldest
        }
        try:
            keep_track_of_tweets_collection.insert_one(q)
        except pymongo.errors.DuplicateKeyError:
            print("attempted dublicate entry")

    else:
        try:
            keep_track_of_tweets_collection.update_one({"_id": user_id},
                                                       {"$set": {"count": all_tweet_count, "oldest": oldest}})
        except:
            print("i was unable to update the keep_track_of_tweets_collection ")
    return tweets_data



def find_and_add_friends_of_user(entrepreneur_id, sample_of_friends, friend_collection,
                          relationship_collection, code_for_friendship,code_for_api):
    api=None
    if(code_for_api==1):
        auth = tweepy.OAuthHandler(consumer_key1, consumer_secret_key1)
        auth.set_access_token(access_tokken1, access_tokken_secret1)
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True,
                         parser=tweepy.parsers.JSONParser())
    else:
        auth = tweepy.OAuthHandler(consumer_key2, consumer_secret_key2)
        auth.set_access_token(access_tokken2, access_tokken_secret2)
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True,
                         parser=tweepy.parsers.JSONParser())
    for friends_id in sample_of_friends:
        # check an en idi mesa
        find_and_add_friend_of_user(friends_id=friends_id,entrepreneur_id=entrepreneur_id,friend_collection=friend_collection,relationship_collection=relationship_collection,code=code_for_friendship,api=api)

def find_and_add_friend_of_user(friends_id, entrepreneur_id, friend_collection,
                                     relationship_collection, code, api):
    # arxika elenxi oti den exo idi get_user afto ton user
    if (friend_collection.find_one({"_id": friends_id}) == None):
        try:
            # an oxi katevazo to xristi
            friend = api.get_user(user_id=friends_id)
            friend = dict(friend)
            friend["_id"] = friend.pop("id")
            try:
                # prospatho na ton valo sto database
                friend_collection.insert_one(friend)
            except WriteError as e:
                print("i could insert the user in database ", friends_id)
                return
        except tweepy.error.TweepError as e:
            print("didnt find this user ", friends_id)
            return
    else:
        print("this user is already in the database ", friends_id)
    # ite iparxi idi ite oxi dimiourgo ti sxesi me afto ton entrepreneur
    if (code == 1):
        relationship = {
            "_id": {
                "is_followed": entrepreneur_id,
                "is_followed_by": friend["_id"]
            }
        }
    else:
        relationship = {
            "_id": {
                "is_followed": friend["_id"],
                "is_followed_by": entrepreneur_id
            }
        }
    # elenxo ean iparxi idi i sxesi
    if (relationship_collection.find_one({relationship}) != None):
        print("relationship already exists")
    else:
        try:
            # ti prostheto sto db
            relationship_collection.insert_one(relationship)
        except pymongo.errors.DuplicateKeyError:
            # ean den katafero na ti valo vgazo
            print("this relationship is already in databse")
        except pymongo.errors.DuplicateKeyError:
            print("there was a problem inserting reletionship")



def create_entrepreneurs_processes(entrepreneurs,followers_collection,following_collection,relationship_collection,
             unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,
             keep_track_of_tweets_collection,max_tweets_per_user,
             max_friends_per_user,starting_index, finishing_index):


    index=starting_index
    pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
    while(index<finishing_index):
        entrepreneur=entrepreneurs[index]

        random_sample_followers=generate_random_sample(entrepreneur["followers_ids"],0,len(entrepreneur["followers_ids"]),max_friends_per_user)
        random_sample_following = generate_random_sample(entrepreneur["following_ids"], 0, len(entrepreneur["following_ids"]),max_friends_per_user)
        arguments=(entrepreneur,random_sample_followers,random_sample_following,followers_collection,following_collection,relationship_collection,
             unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,
             keep_track_of_tweets_collection,max_tweets_per_user)
        pool.apply_async(create_entrepreneur_process, args=arguments)
        index=index+1

    pool.close()
    pool.join()



def create_entrepreneur_process(entrepreneur,random_sample_followers,random_sample_following,followers_collection,following_collection,relationship_collection,
             unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,
             keep_track_of_tweets_collection,max_tweets_per_user):
    # code=1->followers
    code_for_followers = 1
    # code=2->Followings
    code_for_followings = 2
    #code_for_api1
    code_for_api_1=1
    #code_for_api2
    code_for_api_2=2


    args1 = (random_sample_followers, unprocessed_tweets_of_friends_collection, processed_tweets_of_friends_collection,
             keep_track_of_tweets_collection, max_tweets_per_user, code_for_api_1)
    args2 = (entrepreneur["_id"], random_sample_followers, followers_collection, relationship_collection, code_for_followers,
    code_for_api_1)
    args3 = (random_sample_following, unprocessed_tweets_of_friends_collection, processed_tweets_of_friends_collection,
             keep_track_of_tweets_collection, max_tweets_per_user, code_for_api_2)
    args4 = (entrepreneur["_id"], random_sample_following, following_collection, relationship_collection, code_for_followings,
    code_for_api_2)

    pool = multiprocessing.Pool()  # use all available cores, otherwise specify the number you want as an argument
    pool.apply_async(find_and_add_tweets_of_users, args=args1)
    pool.apply_async(find_and_add_friends_of_user, args=args2)
    pool.apply_async(find_and_add_tweets_of_users, args=args3)
    pool.apply_async(find_and_add_friends_of_user, args=args4)

    pool.close()
    pool.join()

def main():
    client = MongoClient('localhost', 27017)
    db = client["testDB"]
    followers_collection=db["Followers"]
    following_collection=db["Following"]
    relationship_collection=db["relationships"]
    processed_tweets_of_friends_collection = db["processed_tweets_friends_and_followers"]
    unprocessed_tweets_of_friends_collection = db["unprocessed_tweets_friends_and_followers"]
    final_sample_collection = db["final_sample"]
    keep_track_of_tweets_collection = db["keep_track_of_tweets"]

    entrepreneurs = get_everything_to_a_list(final_sample_collection, True)
    starting_index=0
    finishing_index=len(entrepreneurs)
    max_tweets_per_user = 600
    max_friends_per_user = 150

    create_entrepreneurs_processes(entrepreneurs, followers_collection, following_collection, relationship_collection,
                                   unprocessed_tweets_of_friends_collection, processed_tweets_of_friends_collection,
                                   keep_track_of_tweets_collection, max_tweets_per_user,
                                   max_friends_per_user, starting_index, finishing_index)

    #turn_screen_name_to_lowercase(collection)
if __name__ == "__main__":
    main()





# def mix(entrepreneurs,followers_collection,following_collection,relationship_collection,
#              unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,
#              keep_track_of_tweets_collection,max_tweets_per_user,
#              max_friends_per_user,starting_index, finishing_index):
#     # code=1->followers
#     code_for_followers = 1
#     # code=2->Followings
#     code_for_followings = 2
#     #code_for_api1
#     code_for_api_1=1
#     #code_for_api2
#     code_for_api_2=2
#
#     index=starting_index
#     while(index<finishing_index):
#         entrepreneur=entrepreneurs[index]
#
#         random_sample_followers=generate_random_sample(entrepreneur["followers_ids"],0,len(entrepreneur["followers_ids"]),max_friends_per_user)
#         random_sample_following = generate_random_sample(entrepreneur["following_ids"], 0, len(entrepreneur["following_ids"]),max_friends_per_user)
#
#         args1=(random_sample_followers,unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,keep_track_of_tweets_collection,max_tweets_per_user,code_for_api_1)
#         args2=(entrepreneur["_id"],random_sample_followers,followers_collection,relationship_collection,code_for_followers,code_for_api_1)
#         args3=(random_sample_following,unprocessed_tweets_of_friends_collection,processed_tweets_of_friends_collection,keep_track_of_tweets_collection,max_tweets_per_user,code_for_api_2)
#         args4=(entrepreneur["_id"],random_sample_following,following_collection,relationship_collection,code_for_followings,code_for_api_2)
#
#
#         p1 = Process(target=find_and_add_tweets_of_users,args=args1)
#         p2 = Process(target=find_and_add_friends_of_user,args=args2)
#         p3 = Process(target=find_and_add_tweets_of_users,args=args3)
#         p4 = Process(target=find_and_add_friends_of_user,args=args4)
#
#         p1.start()
#         p2.start()
#         p3.start()
#         p4.start()
#
#         p1.join()
#         p2.join()
#         p3.join()
#         p4.join()
#
#         index=index+1
#

# def get_all_friends_users(entrepreneurs, friend_collection,
#                           relationship_collection, max_friends_per_user, code):
#     # code=1->followers
#     # code=2->followings
#     for entrepreneur in entrepreneurs:
#         if (code == 1):
#             friends_ids = entrepreneur["followers_ids"]
#         else:
#             friends_ids = entrepreneur["following_ids"]
#
#         random_sample = generate_random_sample(friends_ids,
#                                                0, len(friends_ids), max_friends_per_user)
#
#         for friends_id in random_sample:
#             # check an en idi mesa
#             if (friend_collection.find_one({"_id": friends_id}) == None):
#                 try:
#                     friend = api.get_user(user_id=friends_id)
#                     friend = dict(friend)
#                     friend["_id"] = friend.pop("id")
#                     try:
#                         friend_collection.insert_one(friend)
#                     except WriteError as e:
#                         print("i could insert the user in database ", friends_id)
#                     if (code == 1):
#                         relationship = {
#                             "is_followed": entrepreneur["_id"],
#                             "is_followed_by": friend["_id"]
#                         }
#                     else:
#                         relationship = {
#                             "is_followed": friend["_id"],
#                             "is_followed_by": entrepreneur["_id"]
#                         }
#                     try:
#                         relationship_collection.insert_one(relationship)
#                     except pymongo.errors.DuplicateKeyError:
#                         print("this relationship is already in databse")
#                         print("removing following", friend["_id"])
#                         relationship_collection.remove({"_id": friend["_id"]})
#                     except pymongo.errors.DuplicateKeyError:
#                         print("there was a problem inserting reletionship")
#                         print("removing following", friend["_id"])
#                         relationship_collection.remove({"_id": friend["_id"]})
#
#                 except tweepy.error.TweepError as e:
#                     print("didnt find this user ", friends_id)
#             else:
#                 print("this user is already in the database ", friends_id)
#                 continue

# def find_and_add_all_tweets_of_user(user_id: int, unprocessed, processed, keep_track_of_tweets_collection,
#                                     max_tweets_per_user=3240):
#     # Twitter only allows access to a users most recent 3240 tweets with this method
#
#     temp = keep_track_of_tweets_collection.find_one({"_id": user_id})
#
#     if (temp == None):
#         all_tweet_count = 0
#     else:
#         all_tweet_count = temp["count"]
#
#     # initialize a list to hold all the tweepy Tweets and the tweet data, soon to be the df
#     tweets_data = []
#     all_tweets = []
#
#     # make initial request for most recent tweets (200 is the maximum allowed count)
#     try:
#         if (temp == None or temp["oldest"] == -1):
#             new_tweets = api.user_timeline(user_id=user_id, count=200)
#         else:
#             new_tweets = api.user_timeline(user_id=user_id, count=200, max_id=temp["oldest"])
#     except:
#         print("unable to get tweets for user id ", user_id)
#         return tweets_data
#     all_tweets = all_tweets + new_tweets
#     all_tweet_count = all_tweet_count + len(new_tweets)
#     # process the first tweets and add them to the tweets_data
#     processed_tweets = process_these_tweets(new_tweets)
#     tweets_data = tweets_data + processed_tweets
#     for tweet in new_tweets:
#         tweet["_id"] = tweet.pop("id")
#
#     oldest = -1
#
#     # save the id of the oldest tweet less one
#     if (len(new_tweets) > 0):
#         oldest = new_tweets[-1]["_id"] - 1
#
#     try:
#         if (len(new_tweets) > 0):
#             unprocessed.insert_many(new_tweets)
#         if (len(processed_tweets) > 0):
#             processed.insert_many(processed_tweets)
#     except:
#         print("i was unable to save evrything to the database")
#
#     # keep grabbing tweets until there are no tweets left to grab
#     while ((len(new_tweets) > 0) and all_tweet_count < max_tweets_per_user) > 0:
#         print(f"getting tweets before {oldest}")
#         # all subsiquent requests use the max_id param to prevent duplicates
#         try:
#             new_tweets = api.user_timeline(user_id=user_id, count=200, max_id=oldest)
#         except:
#             print("unable to get the rest of the tweets for user id  ", user_id)
#             break
#         all_tweets = all_tweets + new_tweets
#         all_tweet_count = all_tweet_count + len(new_tweets)
#         # process the tweets and add them to the tweets_data
#         processed_tweets = process_these_tweets(new_tweets)
#         tweets_data = tweets_data + processed_tweets
#         for tweet in new_tweets:
#             tweet["_id"] = tweet.pop("id")
#         if (len(new_tweets) > 0):
#             oldest = new_tweets[-1]["_id"] - 1
#         try:
#             if (len(new_tweets) > 0):
#                 unprocessed.insert_many(new_tweets)
#                 # update the id of the oldest tweet less one
#             if (len(processed_tweets) > 0):
#                 processed.insert_many(processed_tweets)
#         except:
#             print("i was unable to save evrything to the database")
#
#         print(f"...{all_tweet_count} tweets downloaded so far")
#     if (temp == None):
#         q = {
#             "_id": user_id,
#             "count": all_tweet_count,
#             "oldest": oldest
#         }
#         try:
#             keep_track_of_tweets_collection.insert_one(q)
#         except pymongo.errors.DuplicateKeyError:
#             print("attempted dublicate entry")
#
#     else:
#         try:
#             keep_track_of_tweets_collection.update_one({"_id": user_id},
#                                                        {"$set": {"count": all_tweet_count, "oldest": oldest}})
#         except:
#             print("i was unable to update the keep_track_of_tweets_collection ")
#     return tweets_data
#
# def get_all_tweets_from_not_entrepreneurs(entrepreneurs,unprocessed,processed,final_sample_collection,
#         keep_track_of_tweets_collection,max_tweets_per_user=3240,max_friends_per_user=5000):
#     misoi=math.ceil(len(entrepreneurs) / 2)
#     for i,entrepreneur in enumerate(entrepreneurs):
#
#         if(i>=misoi):
#             print("i>msoi ",i)
#             break;
#
#         random_sample_followers=generate_random_sample(entrepreneur["followers_ids"],0,len(entrepreneur["followers_ids"]),max_friends_per_user)
#         random_sample_following = generate_random_sample(entrepreneur["following_ids"], 0, len(entrepreneur["following_ids"]),max_friends_per_user)
#
#         for follower_id in random_sample_followers:
#             find_and_add_all_tweets_of_user(follower_id,unprocessed,processed,keep_track_of_tweets_collection,max_tweets_per_user)
#         for following_id in random_sample_following:
#             find_and_add_all_tweets_of_user(following_id,unprocessed,processed,keep_track_of_tweets_collection,max_tweets_per_user)

