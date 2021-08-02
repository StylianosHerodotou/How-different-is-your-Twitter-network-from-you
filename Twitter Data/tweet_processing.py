import tweepy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

from collect_data_from_twitter import get_all_followers_of_user, get_all_friends_of_user
from tweet_text_preprocessing import clean_tweet
from statistics import mean,stdev


def process_these_tweets(tweets, column_keys) -> list:
    keys = column_keys
    tweets_data = []
    # gia kathe tweeet
    for i, status in enumerate(tweets):
        status_dict = dict(status)
        print(status.keys())

        # ta keys p en eshi apefthias pio kato.
        user_id = status["user"]["id"]
        user = status["user"]["screen_name"]
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

        # neg=negative, neu=neutral , pos=positive compound vasika
        # positive sentiment: compound score >= 0.05
        # neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
        # negative sentiment: compound score <= -0.05
        # -1 (most extreme negative) and +1 (most extreme positive).
        # me to for vali ta columns p en mesti lista colums.
        single_tweet_data = {"_id": inti,
                             "user": user,
                             "user_id": user_id,
                             "original_text": original_text,
                             "clean_text": clean_text,
                             "polarity": polarity,
                             "subjectivity": subjectivity,
                             "hashtags": hashtags,
                             "user_mentions": user_mentions,
                             "sentiment": sentiment
                             }
        for k in keys:
            try:
                v_type = type(status_dict[k])
            except:
                v_type = None
                print(i, k, " none")
            if v_type != None:
                single_tweet_data[k] = status_dict[k]
        # dame telioni to for.
        tweets_data.append(single_tweet_data)

    return tweets_data


def process_this_user(user_screen_name: str,collection) -> dict:
    user = collection.find_one({"screen_name": user_screen_name.lower()})
    follower_count = user.followers_count
    friends_count = user.friends_count
    name = user.name
    user_id = user.id
    followers_ids = get_all_followers_of_user(user_screen_name)
    following_ids = get_all_friends_of_user(user_screen_name)

    ans = {"user_id": user_id,
           "name": name,
           "friends_count": friends_count,
           "follower_count": follower_count,
           "followers_ids": followers_ids,
           "following_ids": following_ids,
           }
    return ans

def process_list_of_users(list_of_users) -> list:
    ans = []
    for user in list_of_users:
        ans.append(process_this_user(user))
    return ans

def find_average_of_list(lista):
    return mean(lista)
def find_standard_deviation(lista):
    return stdev(lista)


def detect_outlier_with_z_value(data_1):
    outliers = []
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for i, y in enumerate(data_1):
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append((i, y))
    return (outliers)


def find_only_int_and_float(df):
    s = set()
    dts = dict(df.dtypes)
    for key in dts.keys():
        if ((dts[key].name == "int64") or (dts[key].name == "float64")):
            #             print(dts[key])
            s.add(key)

    lista = list(s)
    new_df = df[lista].copy()
    return new_df

def IQR_score_with_df(df):
    new_df=find_only_int_and_float(df)
    Q1 = new_df.quantile(0.25)
    Q3 = new_df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    print(new_df < (Q1 - 1.5 * IQR)) or (new_df > (Q3 + 1.5 * IQR))

def detect_outliers_with_IQR(lista):
    outliers = []
    sorted(lista)
    q1, q3 = np.percentile(lista, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for item in lista:
        if (item < lower_bound or item > upper_bound):
            outliers.append(item)
    return outliers

