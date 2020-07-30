#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt 
from pymongo import MongoClient
import numpy as np
import tweepy
import json
import pandas as pd
import csv
import re #regular expression
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
from datetime import date
from datetime import datetime


# In[ ]:


# path=r"C:\Users\35796\Downloads\DATASET_Crunchbase_Founders_with_Twitter.csv"
# df=pd.read_csv(path, delimiter="\t")


# In[ ]:


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def get_current_date():
    temp=datetime.today().strftime('%d-%m-%Y')
    temp=temp.split("-")
    ans_date=(int(temp[0]),int(temp[1]),int(temp[2]))
    return ans_date

def get_days_so_far(today, then):
    d_today=date(today[2],today[1],today[0])
    d_then=date(then[2],then[1],then[0])
    days=d_today-d_then
    return days.days

def find_number_of_posts_per_week(number_of_posts, created_at):
    then=find_date(created_at)
    today=get_current_date()
    days_between=get_days_so_far(today,then)
    ans=number_of_posts/days_between
    return ans

import re
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


def get_avg_sentiment_of_user(user_id):

    all_tweeets_of_user=collection.find({"user":{"id":user_id}})
    count=0
    for i,tweet in enumerate(all_tweeets_of_user):
        sum_sentiment=sum_sentiment+tweet["sentiment"]
        count=i+1

    return sum_sentiment/count


    



def find_date(date_str:str):
    months={ 'Jan':1,
        "Feb":2,
        "Mar":3,
        "Apr":4,
        "May": 5,
        "Jun":6,
        "Jul":7,
        "Aug":8,
        "Sep":9,
        "Oct":10,
        "Nov":11,
        "Dec":12}
    
    splitted=date_str.split()
    month=months[splitted[1]]
    day=splitted[2]
    year=splitted[-1]
    try:
        ans_date=(int(day),month,int(year))
    except:
        ans_date=None
    return ans_date

def find_followers_following_ratio(follower_count, following_count):
    if(following_count==0):
        return 0
    else:
        return follower_count/following_count


# In[ ]:


import re
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

def find_number_of_posts_per_week(number_of_posts, created_at):
    then=find_date(created_at)
    today=get_current_date()
    days_between=get_days_so_far(today,then)
    ans=number_of_posts/days_between
    return ans

def clean_tweet(dirty:str)->str:
    stop_words = set(stopwords.words('english'))

    
    #gia na eshi ta new lines
    tweet = dirty.replace('\n', ' ').replace('\r', '')
    #Convert text to lowercase
    tweet = tweet.lower()
#     print("lower",tweet)
    #removes punctuation
    tweet=remove_punctuation(tweet)
#     print("punctuation",tweet)
    #remove contractions
    tweet=removeContractions(tweet)
#     print("contractions",tweet)
    #clean this shit
    tweet=p.clean(tweet)
#     print("cleaned it",tweet)
    #after tweepy preprocessing the colon symbol left remain after removing mentions
    
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
#     print("ascii char",tweet)
    tweet = re.sub(r':', '', tweet)
#     print(tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
#     print(tweet)
#remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    #print("emojies",tweet)
#remove numbers
    tweet = re.sub(r"\d+", "", tweet)
   # print("remove numbers",tweet)
    
#lemmatize text
    tweet=lemmatize_text(tweet)
   # print("lem",tweet)

#token word
    word_tokens = word_tokenize(tweet)
#remove stopwords
    clean=[]
    w_t=[]
    for i in word_tokens:
        if i not in stop_words:
            w_t.append(i)
    word_tokens=w_t
        
        
  #  print("stop words",' '.join(word_tokens))
#looping through conditions
    for w in word_tokens:
#check tokens against stop words , emoticons and punctuations
        if w not in emoticons :
            clean.append(w)
    return ' '.join(clean)
    


# In[ ]:


def get_avg_sentiment_of_user(user_id):

    all_tweeets_of_user=collection.find({"user":{"id":user_id}})
    count=0
    for i,tweet in enumerate(all_tweeets_of_user):
        sum_sentiment=sum_sentiment+tweet["sentiment"]
        count=i+1

    return sum_sentiment/count


    


# In[ ]:


def process_these_tweets(tweets,processed_tweets_collection):
    
    processed_tweets=[]
    for tweet in tweets:
        if(processed_tweets_collection.find_one({"_id":tweet["_id"]})!=None):
            continue
        processed_tweet=process_this_tweet(tweet)
        processed_tweets.append(processed_tweet)
    return processed_tweets

    


# In[ ]:


def process_this_tweet(tweet):
    status=tweet
    status_dict=dict(status)
    #ta keys p en eshi apefthias pio kato.
    user_id=status_dict["user"]["id"]
    user=status_dict["user"]["screen_name"]
    original_text=status_dict["text"]
    clean_text=clean_tweet(original_text)
    # find polarity and subjectiviy
    blob = TextBlob(clean_text)
    Sentiment = blob.sentiment    
    polarity = Sentiment.polarity
    subjectivity = Sentiment.subjectivity
    #find hashtags
    hashtags=[]
    lod_for_hashtags=status_dict["entities"]["hashtags"]
    for dic in lod_for_hashtags:
        hashtags.append(dic["text"])
    #find user_mentions
    user_mentions=[]
    lod_for_user_mentions=status_dict["entities"]["user_mentions"]
    for dic in lod_for_user_mentions:
        user_mentions.append(dic["screen_name"])
    #find sentiment
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(original_text)
    sentiment=vs["compound"] # mono tuto valo ala maybe ena theli je ta alla 3. tra na dume. 
    positive=vs["pos"]
    negative=vs["neg"]
    neutral=vs["neu"]
    #find id
    inti=status_dict["_id"]
    is_quote=status_dict["is_quote_status"]
    retweet_count=status_dict["retweet_count"]
    favorite_count=status_dict["favorite_count"]
    is_reply=None
    if(status_dict["in_reply_to_screen_name"]!=None):
        is_reply=True
    else:
        is_reply=False
    

    #neg=negative, neu=neutral , pos=positive compound vasika 
    # positive sentiment: compound score >= 0.05
    # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    # negative sentiment: compound score <= -0.05
    #-1 (most extreme negative) and +1 (most extreme positive).
    #me to for vali ta columns p en mesti lista colums.
    ans={ "_id":inti,
                                "user":user,
                               "user_id": user_id,
                               "original_text":original_text,
                               "clean_text": clean_text,
                               "polarity": polarity,
                               "subjectivity": subjectivity,
                               "hashtag_count": len(hashtags),
                               "user_mention_count": len(user_mentions),
                               "sentiment":sentiment,
                           "is_quote":is_quote,
                       "retweet_count":retweet_count,
                       "favorite_count":favorite_count,
                       "is_reply":is_reply
                       
                       
                       
                              }
    return ans
        


# In[ ]:


def process_and_add_tweets_to_database(unpne_tweets_col,processed_tweets_collection,
                    max_tweets=322544,start_from_beggining=False):
    flag=False
    if(start_from_beggining==False):
        count_of_tweets=processed_tweets_collection.estimated_document_count()
    else:
        processed_tweets_collection.delete_many({})
        count_of_tweets=0
    if(max_tweets<count_of_tweets):
        return
    print("count_of_tweets",count_of_tweets)
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
        if(count_of_tweets>max_tweets):
            flag=True
            break
        processed_tweets=process_these_tweets(unpne_tweets,processed_tweets_collection)
        try:
            processed_tweets_collection.insert_many(processed_tweets)
            print("i have just added ",count_of_tweets," tweets to the database")
        except:
            print("i could not add all of them")
        


# In[ ]:


def process_user(user:dict, processed_tweets_collection,code,
             processed_friends_collection=None,
            relationship_collection=None,
            df=None,
            max_number_of_friends=150,
                max_tweet_count=600): 
    print("in process_user_function")
    all_his_tweets=processed_tweets_collection.find({"user_id":user["_id"]}).limit(max_tweet_count)
    processed_tweets_list=list(all_his_tweets)
    print("processing user",user["screen_name"])
    print("his tweets are ", len(processed_tweets_list))
    his_tweets=processed_tweets_list
    twitter_id=user["_id"]
    screen_name=user["screen_name"]
    has_URL=None
    if(user["url"]!=None):
        has_URL=True
    else:
        has_URL=False
    is_verified=user["verified"]
    is_default_profile=user["default_profile"]
    is_default_profile_image=user["default_profile_image"]
    is_protected=user["protected"]
    name_distance=levenshtein(screen_name,user["name"])
    description=process_description(user["description"])
    description_polarity=description["polarity"]
    description_subjectivity=description["subjectivity"]
    description_hashtag_count=description["hashtag_count"]
    description_user_mention_count= description["user_mention_count"]
    description_sentiment= description["sentiment"]
    followers_count=user["followers_count"]
    friends_count=user["friends_count"]
    listed_count=user["listed_count"]
    statuses_count=user["statuses_count"]
    statuses_count_per_week=find_number_of_posts_per_week(statuses_count,user["created_at"])
    followers_following_ratio=find_followers_following_ratio(followers_count,friends_count)
    
    sum_sentiment=0
    sum_polarity_of_user=0
    sum_subjectivity=0
    hashtags_count=0
    user_mentions_count=0
    quote_count=0
    sum_retweet_count=0
    sum_favorite_count=0
    reply_count=0
    for tweet in his_tweets:
        sum_sentiment=sum_sentiment+tweet['sentiment']
        sum_polarity_of_user=sum_polarity_of_user+tweet["polarity"]
        sum_subjectivity=sum_subjectivity+tweet['subjectivity']
        sum_retweet_count=sum_retweet_count+tweet["retweet_count"]
        sum_favorite_count=sum_favorite_count+tweet["favorite_count"]
        hashtags_count=hashtags_count+tweet["hashtag_count"]
        user_mentions_count=user_mentions_count+tweet["user_mention_count"]
        if(tweet["is_quote"]==True):
            quote_count=quote_count+1
        if(tweet["is_reply"]==True):
            reply_count=reply_count+1
            
    
    posa_tweets_exi=len(his_tweets)
    if(posa_tweets_exi!=0):
        avg_polarity_of_user=sum_polarity_of_user/posa_tweets_exi
        avg_subjectivity_of_user=sum_subjectivity/posa_tweets_exi
        avg_hashtag_usage=hashtags_count/posa_tweets_exi
        avg_mention_usage=user_mentions_count/posa_tweets_exi
        avg_sentiment_of_user=sum_sentiment/posa_tweets_exi
        quote_count_in_tweets=quote_count
        avg_quote_usage=quote_count/posa_tweets_exi
        avg_retweets_count=sum_retweet_count/posa_tweets_exi
        avg_favorite_count=sum_favorite_count/posa_tweets_exi
        reply_ratio=reply_count/posa_tweets_exi
        favorites_count=sum_favorite_count
    else:
        avg_polarity_of_user=0
        avg_subjectivity_of_user=0
        avg_hashtag_usage=0
        avg_mention_usage=0
        avg_sentiment_of_user=0
        quote_count_in_tweets=0
        avg_quote_usage=0
        avg_retweets_count=0
        avg_favorite_count=0
        reply_ratio=0
        favorites_count=0
    user_dict={
        "_id":twitter_id,
                            "screen_name":screen_name,
                            "has_URL":has_URL,
                            "is_verified":is_verified,
                            "is_default_profile":is_default_profile,
                            "is_default_profile_image":is_default_profile_image,
                            "is_protected":is_protected,
                            "name_distance":name_distance,
                            "description_polarity":description_polarity,
                            "description_subjectivity":description_subjectivity,
                            "description_hashtag_count":description_hashtag_count,
                            "description_user_mention_count":description_user_mention_count,
                            "description_sentiment":description_sentiment,
                            "followers_count":followers_count,
                            "friends_count":friends_count,
                            "listed_count":listed_count,
                            "favorites_count":favorites_count,
                            "statuses_count":statuses_count,
                            "statuses_count_per_week":statuses_count_per_week,
                            "followers_following_ratio":followers_following_ratio,
                            "avg_polarity_of_user":avg_polarity_of_user,
                            "avg_subjectivity_of_user":avg_subjectivity_of_user,
                            "avg_hashtag_usage":avg_hashtag_usage,
                            "avg_mention_usage":avg_mention_usage,
                            "avg_sentiment_of_user":avg_sentiment_of_user,
                            "quote_count_in_tweets":quote_count_in_tweets,
                            "avg_quote_usage":avg_quote_usage,
                            "avg_retweets_count":avg_retweets_count,
                            "avg_favorite_count":avg_favorite_count,
                            "reply_ratio":reply_ratio
    }
    if(code==1):   #gia followers
        print("follows")
        user_dict["status"]="follower"
    elif (code==2):#gia followings
        print("following")
        user_dict["status"]="following"
    else: #gia final_sample
        user_dict["followers_ids"]=user["followers_ids"]
        user_dict["following_ids"]=user["following_ids"]
        follower_info=find_information_from_network_for_user(user,
        relationship_collection,processed_friends_collection,1,max_number_of_friends
        )
        following_info=find_information_from_network_for_user(user,
        relationship_collection,processed_friends_collection,2,max_number_of_friends
        )
        

        temp= {**follower_info, **following_info}
        temp2={ **temp,**user_dict}
        user_dict=temp2
        temp=df[df["twitter_username"]==user["screen_name"]]
        q=temp["num_funding_rounds"].values[0]
        if(q>0):
            user_dict["got_funding"]=1
        else:
            user_dict["got_funding"]=0
   
    print("processed ",user_dict["_id"])

    return user_dict
        
    

    


# In[ ]:


def process_users(users,
    processed_tweets_collection,
    processed_users_collection,
    code,
    relationship_collection=None,
    processed_friends_collection=None,
    df=None,
    max_tweet_count=600,
    max_number_of_friends=150):
    processed_users=[]
    for user in users:
        
        if(processed_users_collection.find({"_id":user["_id"]})!=None):
            processed_user=process_user(user,
                                        processed_tweets_collection,code,max_number_of_friends=max_number_of_friends,
                                    processed_friends_collection=processed_friends_collection,
                                       relationship_collection=relationship_collection, df=df,max_tweet_count=max_tweet_count)
            if(processed_user!=None):
                processed_users.append(processed_user)
        else:
            print("user already processed")
            continue                   
    return processed_users
        


# In[ ]:


def find_number_of_posts_per_week(number_of_posts, created_at):
    then=find_date(created_at)
    today=get_current_date()
    days_between=get_days_so_far(today,then)
    ans=number_of_posts/days_between
    return ans


# In[ ]:


def find_information_from_network_for_user(user,relationship_collection,friend_collection,
                                        second_code,max_friends_count=150):
    all_friends=[]
    if(second_code==1):
        q=relationship_collection.find({"_id.is_followed":user["_id"]}).limit(max_friends_count)
        if(q!=None):
            relationships=list(q)
        else:
            relationships=[]
        ids=[]
        for relationship in relationships:
            ids.append(relationship["_id"]["is_followed_by"])
        q=friend_collection.find({"_id": {"$in": ids}})
        if(q!=None):
            all_friends=list(q)
        else:
            all_friends=[]
    else:
        q= relationship_collection.find({"_id.is_followed_by":user["_id"]}).limit(max_friends_count)
        if(q!=None):
            relationships=list(q)
        else:
            relationships=[]
        ids=[]
        for relationship in relationships:
            ids.append(relationship["_id"]["is_followed"])
        q=friend_collection.find({"_id": {"$in": ids}})
        if(q!=None):
            all_friends=list(q)
        else:
            all_friends=[]
    count_of_friends=len(all_friends)
    print("count_of_friends",count_of_friends)
    #find info
    count_has_URL=0
    count_is_verified=0;
    count_is_default_profile=0
    count_is_default_profile_image=0
    count_is_protected=0
    sum_name_distance=0
    sum_description_polarity=0
    sum_description_subjectivity=0
    sum_description_hashtag_count=0
    sum_description_user_mention_count=0
    sum_description_sentiment=0
    sum_followers_count=0
    sum_friends_count=0
    sum_listed_count=0
    sum_favorites_count=0
    sum_statuses_count_per_week=0
    sum_followers_following_ratio=0
    sum_polarity=0
    sum_subjectivity=0
    sum_hashtag_usage=0
    sum_mention_usage=0
    sum_sentiment=0
    sum_quote_usage=0
    sum_retweets_count=0
    sum_favorite_count=0
    sum_reply_ratio=0
    for friend in all_friends:
        if(friend["has_URL"]==True):
            count_has_URL=count_has_URL+1
        if(friend["is_verified"]==True):
            count_is_verified=count_is_verified+1
        if(friend["is_default_profile"]==True):
            count_is_default_profile=count_is_default_profile+1
        if(friend["is_default_profile_image"]==True):
            count_is_default_profile_image=count_is_default_profile_image+1
        if(friend["is_protected"]==True):
            count_is_protected=count_is_protected+1
        sum_name_distance=sum_name_distance+friend["name_distance"]
        sum_description_polarity=sum_description_polarity+friend["description"]["polarity"]
        sum_description_subjectivity=sum_description_subjectivity+friend["description"]["subjectivity"]
        sum_description_hashtag_count=sum_description_hashtag_count+friend["description"]["hashtag_count"]
        sum_description_user_mention_count=sum_description_user_mention_count+friend["description"]["user_mention_count"]
        sum_description_sentiment= sum_description_sentiment+friend["description"]["sentiment"]
        sum_followers_count=sum_followers_count+friend["followers_count"]
        sum_friends_count=sum_friends_count +friend["friends_count"]
        sum_listed_count=sum_listed_count+friend["listed_count"]
        sum_favorites_count=sum_favorites_count+friend["favorites_count"]
        sum_statuses_count_per_week=sum_statuses_count_per_week+friend["statuses_count_per_week"]
        sum_followers_following_ratio=sum_followers_following_ratio+friend["followers_following_ratio"]
        sum_polarity=sum_polarity+friend["avg_polarity_of_user"]
        sum_subjectivity= sum_subjectivity+friend["avg_subjectivity_of_user"]
        sum_hashtag_usage=sum_hashtag_usage+friend["avg_hashtag_usage"]
        sum_mention_usage=sum_mention_usage+friend["avg_mention_usage"]
        sum_sentiment=sum_sentiment+friend["avg_sentiment_of_user"]
        sum_quote_usage=sum_quote_usage+friend["avg_quote_usage"]
        sum_retweets_count=sum_retweets_count+friend["avg_retweets_count"]
        sum_favorite_count=sum_favorite_count+friend["avg_favorite_count"]
        sum_reply_ratio=sum_reply_ratio+friend["reply_ratio"]
    if(len(all_friends)!=0):
        posa=len(all_friends)
        avg_has_URL=count_has_URL/posa
        avg_is_verified=count_is_verified/posa
        avg_is_default_profile=count_is_default_profile/posa
        avg_is_default_profile_image=count_is_default_profile_image/posa
        avg_is_protected=count_is_protected/posa
        avg_name_distance=sum_name_distance/posa
        avg_description_polarity=sum_description_polarity/posa
        avg_description_subjectivity=sum_description_subjectivity/posa
        avg_description_hashtag_count=sum_description_hashtag_count/posa
        avg_description_user_mention_count=sum_description_user_mention_count/posa
        avg_description_sentiment=sum_description_sentiment/posa
        avg_followers_count=sum_followers_count/posa
        avg_friends_count=sum_friends_count/posa
        avg_listed_count=sum_listed_count/posa
        avg_favorites_count=sum_favorites_count/posa
        avg_statuses_count_per_week=sum_statuses_count_per_week/posa
        avg_followers_following_ratio=sum_followers_following_ratio/posa
        avg_avg_polarity_of_user=sum_polarity/posa
        avg_avg_subjectivity_of_user=sum_subjectivity/posa
        avg_avg_hashtag_usage=sum_hashtag_usage/posa
        avg_avg_mention_usage=sum_mention_usage/posa
        avg_avg_sentiment_of_user=sum_sentiment/posa
        avg_avg_quote_usage=sum_quote_usage/posa
        avg_avg_retweets_count=sum_retweets_count/posa
        avg_avg_favorite_count=sum_favorite_count/posa
        avg_reply_ratio=sum_reply_ratio/posa
    else:
        avg_has_URL=0
        avg_is_verified=0
        avg_is_default_profile=0
        avg_is_default_profile_image=0
        avg_is_protected=0
        avg_name_distance=0
        avg_description_polarity=0
        avg_description_subjectivity=0
        avg_description_hashtag_count=0
        avg_description_user_mention_count=0
        avg_description_sentiment=0
        avg_followers_count=0
        avg_friends_count=0
        avg_listed_count=0
        avg_favorites_count=0
        avg_statuses_count_per_week=0
        avg_followers_following_ratio=0
        avg_avg_polarity_of_user=0
        avg_avg_subjectivity_of_user=0
        avg_avg_hashtag_usage=0
        avg_avg_mention_usage=0
        avg_avg_sentiment_of_user=0
        avg_avg_quote_usage=0
        avg_avg_retweets_count=0
        avg_avg_favorite_count=0
        avg_reply_ratio=0
    ans=None
    if(second_code==1):  
        ans={
"percent_has_URL_followers":avg_has_URL,
"percent_is_verified_followers":avg_is_verified,
"percent_is_default_profile_followers":avg_is_default_profile,
"percent_is_default_profile_followers":avg_is_default_profile_image,
"percent_is_protected_followers":avg_is_protected,
"avg_name_distance_followers":avg_name_distance,
"avg_description_polarity_followers":avg_description_polarity,
"avg_description_subjectivity_followers":avg_description_subjectivity,
"avg_description_hashtag_count_followers":avg_description_hashtag_count,
"avg_description_user_mention_count_followers":avg_description_user_mention_count,
"avg_description_sentiment_followers":avg_description_sentiment,
"avg_followers_count_followers":avg_followers_count,
"avg_friends_count_followers":avg_friends_count,
"avg_listed_count_followers":avg_listed_count,
"avg_favorites_count_followers":avg_favorites_count,
"avg_statuses_count_per_week_followers":avg_statuses_count_per_week,
"avg_followers_following_ratio_followers":avg_followers_following_ratio,
"avg_polarity_of_followers":avg_avg_polarity_of_user,
"avg_subjectivity_of_followers":avg_avg_subjectivity_of_user,
"avg_hashtag_usage_followers":avg_avg_hashtag_usage,
"avg_mention_usage_followers":avg_avg_mention_usage,
"avg_sentiment_of_followers":avg_avg_sentiment_of_user,
"avg_quote_usage_followers":avg_avg_quote_usage,
"avg_retweets_count_followers":avg_avg_retweets_count,
"avg_favorite_count_followers":avg_avg_favorite_count,
"avg_reply_ratio_followers":avg_reply_ratio
            }
    else:
        ans={ 
"percent_has_URL_following":avg_has_URL,
"percent_is_verified_following":avg_is_verified,
"percent_is_default_profile_following":avg_is_default_profile,
"percent_is_default_profile_following":avg_is_default_profile_image,
"percent_is_protected_following":avg_is_protected,
"avg_name_distance_following":avg_name_distance,
"avg_description_polarity_following":avg_description_polarity,
"avg_description_subjectivity_following":avg_description_subjectivity,
"avg_description_hashtag_count_following":avg_description_hashtag_count,
"avg_description_user_mention_count_following":avg_description_user_mention_count,
"avg_description_sentiment_following":avg_description_sentiment,
"avg_followers_count_following":avg_followers_count,
"avg_friends_count_following":avg_friends_count,
"avg_listed_count_following":avg_listed_count,
"avg_favorites_count_following":avg_favorites_count,
"avg_statuses_count_per_week_following":avg_statuses_count_per_week,
"avg_followers_following_ratio_following":avg_followers_following_ratio,
"avg_polarity_of_following":avg_avg_polarity_of_user,
"avg_subjectivity_of_following":avg_avg_subjectivity_of_user,
"avg_hashtag_usage_following":avg_avg_hashtag_usage,
"avg_mention_usage_following":avg_avg_mention_usage,
"avg_sentiment_of_following":avg_avg_sentiment_of_user,
"avg_quote_usage_following":avg_avg_quote_usage,
"avg_retweets_count_following":avg_avg_retweets_count,
"avg_favorite_count_following":avg_avg_favorite_count,
"avg_reply_ratio_following":avg_reply_ratio
        }
    return ans
        
            
        


# In[ ]:


def process_and_add_users_to_database(
    unprocessed_user_collection,
    processed_tweets_collection,
    processed_user_collection,
    code,
    relationship_collection=None,
    processed_friends_collection=None,
    df=None,
    number_of_users_to_process=1000,
    max_number_of_tweets=600,
    max_number_of_friends=150,
    start_from_beggining=False,
    delete_documents=False,
    per_time=100
    ):
    flag=False
    if(start_from_beggining==False):
        count_of_processed_users=processed_user_collection.estimated_document_count()
    else:
        count_of_processed_users=0
    if(delete_documents==True):
        processed_user_collection.delete_many({})
    if(per_time>number_of_users_to_process):
        per_time=number_of_users_to_process
    while(flag!=True):
        users=unprocessed_user_collection.find({}).skip(count_of_processed_users).limit(per_time)
        if(users==None or count_of_processed_users>= number_of_users_to_process):
            flag=True
            break
        try:
            users=list(users)
            print("i have batch of users of lenght",len(users))
        except:
            continue
        count_of_processed_users=count_of_processed_users+len(users)
        new_processed_users=process_users(users,processed_tweets_collection,
                                                   processed_user_collection,code,relationship_collection,
                                          processed_friends_collection,df,
                                          max_number_of_tweets,max_number_of_friends)
        
        try:
            processed_user_collection.insert_many(new_processed_users)
        except:
            print("i could not add all of them")
        


# In[ ]:


def get_data_ready(number_of_users_to_process=1000,
    max_number_of_tweets=600,
    max_number_of_friends=150,
    start_from_beggining=False,
    process_tweets=False,
    start_from_beggining_tweets=False,
    delete_documents=False,
    do_friends=True,
    do_entrepreneurs=True,
    per_time=100):
    
    code_for_followers=1
    code_for_followings=2
    code_for_entrepreneurs=3
    path=r"C:\Users\35796\Downloads\DATASET_Crunchbase_Founders_with_Twitter.csv"
    df=pd.read_csv(path, delimiter="\t")
    client = MongoClient('10.16.3.55', 27017)
    db=client["testDB"]

    
    unprocessed_followers=db["w_followers"]
    unprocessed_followings=db["w_followings"]
    processed_users_friends=db["w_processed_users_friends"]
    unprocessed_tweets_friends=db["unprocessed_tweets_friends_and_followers"]
    processed_tweets_friends=db["processed_tweets_friends_and_followers"]
    
    
    Entrepreneurs=db["final_sample"]
    processed_entrepreneurs=db["processed_entrepreneurs"]
    processed_tweets_entrepreneurs=db["processed_tweets_entrepreneurs"]
    unprocessed_tweets_entrepreneurs=db["unprocessed_tweets_entrepreneurs"]
    
    relationship_collection=db["relationships"]
    
    if(process_tweets==True):
        process_and_add_tweets_to_database(unprocessed_tweets_friends,
                                   processed_users_friends ,
                                           start_from_beggining_tweets)
        
        process_and_add_tweets_to_database(unprocessed_tweets_entrepreneurs,
                                   processed_tweets_entrepreneurs ,
                                           start_from_beggining_tweets)

        
    if(do_friends==True):
#         #for followers
#         print("doing followers")
#         process_and_add_users_to_database(
#         unprocessed_followers,
#         processed_tweets_friends,
#         processed_users_friends,
#         code_for_followers,
#             processed_friends_collection=processed_users_friends,
#         number_of_users_to_process=number_of_users_to_process,
#         df=df,
#         max_number_of_tweets=max_number_of_tweets,
#         max_number_of_friends=max_number_of_friends,
#             delete_documents=False,
#         start_from_beggining=start_from_beggining,
#         per_time=per_time)

        #for followings
        print("doing followings")
        process_and_add_users_to_database(
        unprocessed_followings,
        processed_tweets_friends,
        processed_users_friends,
        code_for_followings,
        processed_friends_collection=processed_users_friends,
        number_of_users_to_process=number_of_users_to_process,
            df=df,
        max_number_of_tweets=max_number_of_tweets,
        max_number_of_friends=max_number_of_friends,
        start_from_beggining=start_from_beggining,
        delete_documents=False,
        per_time=per_time)
    if(do_entrepreneurs==True):
        #for Entrepreneurs
        print("doing entrepreneurs")
        process_and_add_users_to_database(
        Entrepreneurs,
        processed_tweets_entrepreneurs,
        processed_entrepreneurs,
        code_for_entrepreneurs,
        relationship_collection=relationship_collection,
        processed_friends_collection=processed_users_friends,
        number_of_users_to_process=number_of_users_to_process,
            df=df,
        max_number_of_tweets=max_number_of_tweets,
        max_number_of_friends=max_number_of_friends,
        start_from_beggining=start_from_beggining,
            delete_documents=False,
        per_time=per_time)


# In[ ]:


get_data_ready(number_of_users_to_process=150,
    max_number_of_tweets=600,
    max_number_of_friends=150,
    start_from_beggining=True,
    process_tweets=False,
    start_from_beggining_tweets=True,
    delete_documents=False,
    do_friends=True,
    do_entrepreneurs=True,
    per_time=5)


# In[ ]:


processed_friends_collection.find({"_id":user["_id"]})


# In[ ]:


path=r"C:\Users\35796\Downloads\DATASET_Crunchbase_Founders_with_Twitter.csv"
df=pd.read_csv(path, delimiter="\t")
client = MongoClient('10.16.3.55', 27017)
db=client["testDB"]
collection=db["Starting_dataset"]


# In[25]:


temp=df.to_dict("records")


# In[28]:


collection.insert_many(temp)


# In[ ]:




