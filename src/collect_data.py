import yfinance as yf
import os
import requests
import pandas as pd
import datetime
import time

def get_price_data(ticker, start_date, end_date):
    """
    Fetch historical stock price data for a given ticker symbol.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD'
    Returns:
        pandas.DataFrame: DataFrame containing historical stock prices.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df.reset_index()
    df['Date'] = df['Date'].dt.date
    df.columns = df.columns.droplevel(1)
    return df

def merge_sentiment_price(sentiment_df, price_df):
    """
    Merge sentiment scores with stock price data on the date.
    Args:
        sentiment_df (pandas.DataFrame): DataFrame containing sentiment scores.
        price_df (pandas.DataFrame): DataFrame containing stock prices.
    Returns:
        pandas.DataFrame: Merged DataFrame with sentiment scores and stock prices.
    """
    # Make copies to avoid modifying original DataFrames
    sentiment_df = sentiment_df.copy()
    price_df = price_df.copy()

    # fix multiindex issue
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    
    # Ensure both date columns are in the same format
    if 'Date' in price_df.columns:
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None).dt.normalize()
    
    if 'publishedAt' in sentiment_df.columns:
        # Handle different possible formats for published_date
        if sentiment_df['publishedAt'].dtype == 'object':
            sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt']).dt.tz_localize(None).dt.normalize()
        elif 'datetime' in str(sentiment_df['publishedAt'].dtype):
            sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt']).dt.tz_localize(None).dt.normalize()


    
    # Group sentiment by date and calculate daily averages
    daily_sentiment = sentiment_df.groupby("publishedAt")[["positive", "neutral", "negative"]].mean().reset_index()
    
    # Merge on date columns
    merged_df = pd.merge(
        price_df, 
        daily_sentiment, 
        left_on="Date", 
        right_on="publishedAt", 
        how="left"
    )
    
    # Clean up - remove duplicate date column
    if 'publishedAt' in merged_df.columns:
        merged_df = merged_df.drop('publishedAt', axis=1)
    
    return merged_df


def load_cached_articles(cache_path):
    """
    Load cached articles from a CSV file if it exists.

    Args:
        cache_path (str): Path to the cached articles CSV file.
    Returns:
        pandas.DataFrame: Cached articles if available, otherwise None.    
    """

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["publishedAt"])
    return None

def save_articles_to_cache(articles_df, cache_path):
    """
    Save articles to a CSV file for caching.
    
    Args:
        articles_df (pandas.DataFrame): DataFrame containing articles.
    """
    articles_df.to_csv(cache_path, index=False)

def get_last_cached_timestamp(cache_path):
    """
    Get the last cached timestamp from the cached articles file.
    Args:
        cache_path (str): Path to the cached articles CSV file.
    Returns:
        datetime: Last cached timestamp or None if cache does not exist.
    """
    if not os.path.exists(cache_path):
        return None
    df = pd.read_csv(cache_path, parse_dates=["publishedAt"])
    return df["publishedAt"].max()

def load_articles(ticker, company_name, start_date, end_date, api_key, max_pages=10):
    """
    Load news articles for a given ticker symbol and company name from thenewsapi.com.
    Args:
        ticker (str): Stock ticker symbol.
        company_name (str): Full name of the company.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        api_key (str): TheNewsAPI key.
        max_pages (int): Maximum number of pages to fetch.
    Returns:
        list: List of dictionaries containing news articles.
    """
    all_articles = []

    # Construct the query and endpoint
    query = f'"{company_name}" | {ticker} | "{company_name} stock"'

    url = "https://api.thenewsapi.com/v1/news/all"

    for page in range(1, max_pages + 1):
        params = {
            "api_token": api_key,
            "search": query,
            "published_after": f"{start_date}T00:00:00",
            "published_before": f"{end_date}T23:59:59",
            "language": "en",
            "page": page,
            "limit": 100,  # max allowed per request
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"[!] Error fetching page {page}: {response.status_code}")
            print(response.text)
            break

        data = response.json()
        articles = data.get("data", [])
        if not articles:
            break

        for article in articles:
            # only take english articles since nlp model is for english
            if article["language"] != "en":
                continue
            all_articles.append({
                "ticker": ticker,
                "publishedAt": article.get("published_at"),
                "title": article.get("title"),
                "description": article.get("description"),
                "snippet": article.get("snippet"),
                "source": article.get("source"),
                "url": article.get("url"),
            })

        time.sleep(1.2)  # Respect API rate limits

    return all_articles

def fetch_new_articles(ticker, company_name, api_key, start_date, cache_path):
    """
    Fetch new articles for a given ticker symbol and company name, checking the cache for the last timestamp.

    Args:
        ticker (str): Stock ticker symbol.
        company_name (str): Full name of the company.
        api_key (str): NewsAPI key.
        cache_path (str): Path to the cached articles CSV file.
        Returns:
        list: List of new articles.
    Returns:
        list: List of new articles fetched from NewsAPI.
    """
    # get the last cached timestamp
    last_timestamp = get_last_cached_timestamp(cache_path)
    # default start date if no cache
    if last_timestamp is not None:
        # start from the next second after the last cached timestamp
        start_date = (pd.to_datetime(last_timestamp) + pd.Timedelta(seconds=1)).isoformat()

    # end date is now, the present
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # fetch new articles
    print(f"[↻] Fetching new articles from {start_date} to {end_date}...")

    # call the load_articles function to get the new articles with the specified parameters
    return load_articles(
        ticker=ticker,
        company_name=company_name,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key
    )

def update_cached_articles(new_articles, cache_path):
    """
    Updates the cached articles with new articles, avoiding duplicates.
    Args:
        new_articles (list): List of new articles to be added.
        cache_path (str): Path to the cached articles CSV file.
    Returns:
        pandas.DataFrame: Updated DataFrame with new articles.
    """
    # create a DataFrame from the new articles
    new_df = pd.DataFrame(new_articles)
    # if we already have prexisting data, load it
    if os.path.exists(cache_path):
        old_df = pd.read_csv(cache_path)
        # combine the old and new DataFrames
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        # drop duplicates based on title and publishedAt
        combined_df.drop_duplicates(subset=["title", "publishedAt"], inplace=True)
    else:
        # if no old data, just use the new DataFrame
        combined_df = new_df

    # save the updated DataFrame to cache
    save_articles_to_cache(combined_df, cache_path)
    # return the combined dataFrame
    return combined_df

def get_or_update_articles(ticker, company_name, api_key, start_date, cache_path):
    """
    Gets or updates articles for a given ticker symbol and company name.
    If cached articles exist, it will load them; otherwise, it will fetch new articles.
    Loads the articles, fetches new ones, then updates cache if necessary
    Args:
        ticker (str): Stock ticker symbol.
        company_name (str): Full name of the company.
        api_key (str): NewsAPI key.
        cache_path (str): Path to the folder where the articles CSV file should be
    Returns:
        pandas.DataFrame: DataFrame containing articles, either from cache or newly fetched.
    """
    
    # load cached articles
    cached_df = load_cached_articles(cache_path)
    # fetch new articles
    new_articles = fetch_new_articles(ticker, company_name, api_key, start_date, cache_path)
    
    # if there are new articles, update the cache
    if new_articles:
        full_df = update_cached_articles(new_articles, cache_path)
    # if no new articles, return the cached DataFrame
    else:
        full_df = cached_df if cached_df is not None else pd.DataFrame()

    return full_df