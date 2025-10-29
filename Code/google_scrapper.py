from google_play_scraper import Sort, reviews_all
import pandas as pd

app_package = 'nz.co.vodafone.android.myaccount'
reviews = reviews_all(
    app_package,
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='nz', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
    filter_score_with=None # defaults to None(all scores)
)

df_reviews = pd.DataFrame(reviews)
df_reviews.to_csv("my_one_nz_google_reviews.csv")
print(df_reviews.head())