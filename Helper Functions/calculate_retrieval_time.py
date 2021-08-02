import math
import random

from pymongo import MongoClient


def generate_n_uniqe_random_integers(starting:int=0,final:int =-1,
                                n=10):
    ans=random.sample(range(starting, final), n)
    return ans

def find_waiting_time(number_of_frineds:int,
request_per_window:int,per_call:int,timeout_time:float )->float:
    number_of_windows=math.ceil(
number_of_frineds/(request_per_window*per_call))
    wait_time=(number_of_windows-1)*timeout_time
    return wait_time

def find_request_time(number_of_friends:int,per_call:int,c:float)->float:
    number_of_requests=math.ceil(number_of_friends/per_call)
    time_for_request=number_of_requests*c
    return time_for_request


def find_time_to_get_following_from_user(screen_name: str,collection) -> float:
    c = 0.05  # constant time p theli gi kateh request
    request_per_window = 15
    per_call = 5000
    time = 0.0
    error = 0.0
    timeout_time = (15 * 60-request_per_window*per_call*c)
    user = collection.find_one({"screen_name": screen_name.lower()})
    if (user != None):
        number_of_friends = user["friends_count"]
        # find time for requets
        time_for_requests = find_request_time(number_of_friends, per_call, c)
        # find time ill be waiting
        wait_time = find_waiting_time(number_of_friends,
                                      request_per_window, per_call, timeout_time)

        final_time = wait_time + time_for_requests + error
        print("number of friends ", number_of_friends,
              "waiting_time ", (wait_time / 60),
              "request time ", time_for_requests / 60,
              "final ", final_time / 60)
        return final_time
    else:
        return 0


def find_time_to_get_followers_from_user(screen_name:str,collection)->float:
    c=0.05      #constant time p theli gi kateh request
    request_per_window=15
    per_call=5000
    time=0.0
    error=0.0
    timeout_time = (15 * 60-request_per_window*per_call*c)
    user=collection.find_one({"screen_name": screen_name.lower()})
    if(user!=None):
        number_of_friends=user[ "followers_count"]
        #find time for requets
        time_for_requests=find_request_time(number_of_friends,per_call,c)
        #find time ill be waiting
        wait_time=find_waiting_time(number_of_friends,
    request_per_window,per_call,timeout_time)

        final_time=wait_time+time_for_requests+error
        print("number of followers ",number_of_friends,
              "waiting_time ",(wait_time/60),
             "request time ", time_for_requests/60,
             "final ", final_time/60)
        return final_time
    else:
        return 0


def find_time_to_get_tweets_from_user(screen_name: str,collection) -> float:
    c = 0.05  # constant time p theli gi kateh request
    request_per_window = 1500
    per_call = 1
    time = 0.0
    error = 0.0
    timeout_time = (15 * 60-request_per_window*per_call*c)
    user = collection.find_one({"screen_name": screen_name.lower()})
    if (user != None):
        number_of_friends = user["statuses_count"]
        # find time for requets
        time_for_requests = find_request_time(number_of_friends, per_call, c)
        # find time ill be waiting
        wait_time = find_waiting_time(number_of_friends,
                                      request_per_window, per_call, timeout_time)

        final_time = wait_time + time_for_requests + error
        print("number of tweets ", number_of_friends,
              "waiting_time ", (wait_time / 60),
              "request time ", time_for_requests / 60,
              "final ", final_time / 60)
        return final_time
    else:
        return 0


def find_time_followers_from_users(users_names: list,collection) -> float:
    all_time = 0.0
    for user_name in users_names:
        time = find_time_to_get_followers_from_user(user_name,collection)
        all_time = all_time + time
    return all_time


def find_time_following_from_users(users_names: list,collection) -> float:
    all_time = 0.0
    for user_name in users_names:
        time = find_time_to_get_following_from_user(user_name,collection)
        all_time = all_time + time
    all_time=all_time+len(users_names)*15*60
    return all_time


def ConvertSectoDay(n):
    day = n // (24 * 3600)

    n = n % (24 * 3600)
    hour = n // 3600

    n %= 3600
    minutes = n // 60

    n %= 60
    seconds = n

    print(day, "days", hour, "hours",
          minutes, "minutes",
          seconds, "seconds")


def find_time_tweets_from_users(users_names: list,collection) -> float:
    all_time = 0.0
    for user_name in users_names:
        time = find_time_to_get_tweets_from_user(user_name,collection)
        all_time = all_time + time
    return all_time

def main():
    client = MongoClient('localhost', 27017)
    db = client["testDB"]
    collection = db['test_small_dataset']
    small_dataset = ["therock", "barbaraoakley", "StephenRCovey",
                     "cduhigg", "GregoryMcKeown", "chris_bailey", "Coursera",
                     "freeCodeCamp", "alegonzalezca", "VancityReynolds", "tuitsdediego", "drchuck"]
    time=find_time_following_from_users(small_dataset,collection)
    ConvertSectoDay(time)


if __name__ == "__main__":
    main()