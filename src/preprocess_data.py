import pandas as pd
import numpy as np
import os
import datetime
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import your existing functions
from collect_data import (
    get_price_data,
    merge_sentiment_price,
    get_or_update_articles,
)

from sentiment import score_sentiment

class StockAnalysisDataPipeline:
    """
    A comprehensive pipeline for collecting stock prices, news articles, 
    sentiment scores, and preparing data for machine learning models.
    """
    
    def __init__(self, ticker: str, company_name: str, api_key: str, 
                 cache_dir: str = "data", start_date: str = "2024-01-01"):
        """
        Initialize the data pipeline.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            company_name (str): Full company name (e.g., 'Apple Inc.')
            api_key (str): NewsAPI key
            cache_dir (str): Directory to store cached data
            start_date (str): Start date for data collection in 'YYYY-MM-DD' format
        """
        self.ticker = ticker
        self.company_name = company_name
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.start_date = start_date
        self.end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define file paths
        self.articles_cache_path = os.path.join(cache_dir, f"{ticker}_articles.csv")
        self.sentiment_cache_path = os.path.join(cache_dir, f"{ticker}_sentiment.csv")
        self.price_cache_path = os.path.join(cache_dir, f"{ticker}_prices.csv")
        self.combined_cache_path = os.path.join(cache_dir, f"{ticker}_combined.csv")
        
    def collect_price_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Collect and cache stock price data.
        
        Args:
            force_refresh (bool): If True, fetch fresh data instead of using cache
            
        Returns:
            pandas.DataFrame: Stock price data
        """
        print(f"[📈] Collecting price data for {self.ticker}...")
        
        # Check if cached price data exists and is recent
        if not force_refresh and os.path.exists(self.price_cache_path):
            cached_prices = pd.read_csv(self.price_cache_path)
            cached_prices['Date'] = pd.to_datetime(cached_prices['Date'])

            # Check if cache is recent (within last day)
            last_cached_date = cached_prices['Date'].max()

            # convert to python date for comparison, vs timestamp from pandas
            last_cached_date = last_cached_date.date()
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
            
            if last_cached_date >= yesterday:
                print(f"[✓] Using cached price data (last updated: {last_cached_date})")
                return cached_prices
        
        # Fetch fresh price data
        price_df = get_price_data(self.ticker, self.start_date, self.end_date)
        
        # Save to cache
        price_df.to_csv(self.price_cache_path, index=False)
        print(f"[✓] Price data collected and cached: {len(price_df)} records")
        
        return price_df
    
    def collect_articles_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Collect and cache news articles data.
        
        Args:
            force_refresh (bool): If True, fetch fresh data instead of using cache
            
        Returns:
            pandas.DataFrame: News articles data
        """
        print(f"[📰] Collecting articles data for {self.ticker}...")
        
        if force_refresh and os.path.exists(self.articles_cache_path):
            os.remove(self.articles_cache_path)
        
        # Use your existing function to get or update articles
        articles_df = get_or_update_articles(
            ticker=self.ticker,
            company_name=self.company_name,
            api_key=self.api_key,
            start_date = self.start_date,
            cache_path=self.articles_cache_path
        )
        
        if len(articles_df) == 0:
            print("[⚠] No articles found!")
            return pd.DataFrame()
        
        # Ensure publishedAt is datetime and create date column
        articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt'])
        articles_df['published_date'] = articles_df['publishedAt'].dt.date
        
        print(f"[✓] Articles data collected: {len(articles_df)} articles")
        return articles_df
    
    def score_articles_sentiment(self, articles_df: pd.DataFrame, 
                               force_refresh: bool = False) -> pd.DataFrame:
        """
        Score sentiment of articles.
        
        Args:
            articles_df (pd.DataFrame): DataFrame containing articles
            force_refresh (bool): If True, recalculate sentiment scores
            
        Returns:
            pandas.DataFrame: Articles with sentiment scores
        """
        print(f"[🎭] Scoring sentiment for articles...")
        
        if len(articles_df) == 0:
            print("[⚠] No articles to score sentiment for!")
            return pd.DataFrame()
        
        # Check if sentiment scores already exist
        if not force_refresh and os.path.exists(self.sentiment_cache_path):
            try:
                sentiment_df = pd.read_csv(self.sentiment_cache_path)
                sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'])
                sentiment_df['published_date'] = sentiment_df['publishedAt'].dt.date
                
                # Check if we have sentiment scores for all articles
                if len(sentiment_df) >= len(articles_df):
                    print(f"[✓] Using cached sentiment scores: {len(sentiment_df)} records")
                    return sentiment_df
            except Exception as e:
                print(f"[⚠] Error loading cached sentiment: {e}")
        
        # Save articles to temporary CSV for sentiment scoring
        temp_articles_path = os.path.join(self.cache_dir, f"temp_{self.ticker}_articles.csv")
        articles_df.to_csv(temp_articles_path, index=False)
        
        # Score sentiment using your existing function
        sentiment_df = score_sentiment(temp_articles_path, save=True, prob_scores=True)
        
        # Clean up temporary file
        if os.path.exists(temp_articles_path):
            os.remove(temp_articles_path)
        
        # Save sentiment scores to cache
        sentiment_df.to_csv(self.sentiment_cache_path, index=False)
        print(f"[✓] Sentiment scores calculated and cached: {len(sentiment_df)} records")
        
        return sentiment_df
    
    def create_combined_dataset(self, price_df: pd.DataFrame, 
                              sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine price data with sentiment scores and create features for ML.
        
        Args:
            price_df (pd.DataFrame): Stock price data
            sentiment_df (pd.DataFrame): Sentiment scores data
            
        Returns:
            pandas.DataFrame: Combined dataset ready for ML
        """
        print(f"[🔗] Combining datasets...")
        
        if len(price_df) == 0 or len(sentiment_df) == 0:
            print("[⚠] Cannot combine datasets - missing price or sentiment data!")
            return pd.DataFrame()
        
        # Merge sentiment with price data
        combined_df = merge_sentiment_price(sentiment_df, price_df)
        
        # Add additional features for ML
        combined_df = self._add_ml_features(combined_df)
        
        # Save combined dataset
        combined_df.to_csv(self.combined_cache_path, index=False)
        print(f"[✓] Combined dataset created: {len(combined_df)} records")
        
        return combined_df
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional features for machine learning.
        
        Args:
            df (pd.DataFrame): Combined dataset
            
        Returns:
            pandas.DataFrame: Dataset with additional ML features
        """
        df = df.copy()

        # make sure data is in floats
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['volume_change'] = df['Volume'].pct_change()
        
        # Calculate returns (percentage change)
        df['returns'] = df['Close'].pct_change()
        
        # Remove first row with NaN return
        df = df.iloc[1:].reset_index(drop=True)

        # Moving averages
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_10'] = df['Close'].rolling(window=10).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        
        # Price relative to moving averages
        df['price_to_ma_5'] = df['Close'] / df['ma_5']
        df['price_to_ma_10'] = df['Close'] / df['ma_10']
        df['price_to_ma_20'] = df['Close'] / df['ma_20']
        
        # Volatility (rolling standard deviation)
        df['volatility_5'] = df['Close'].rolling(window=5).std()
        df['volatility_10'] = df['Close'].rolling(window=10).std()

        # Target variable (next day's price change)
        df['target_next_day_change'] = df['price_change'].shift(-1)
        df['target_next_day_up'] = (df['target_next_day_change'] > 0).astype(int)
        
        # High-low spread
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Days since start
        df['days_since_start'] = (pd.to_datetime(df['Date']) - pd.to_datetime(df['Date']).min()).dt.days
        
        # Day of week
        df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
        
        return df
    
    def run_full_pipeline(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Args:
            force_refresh (bool): If True, refresh all cached data
            
        Returns:
            pandas.DataFrame: Complete dataset ready for ML
        """
        print(f"[🚀] Starting full pipeline for {self.ticker} ({self.company_name})")
        print("=" * 60)
        
        try:
            # Step 1: Collect price data
            price_df = self.collect_price_data(force_refresh=force_refresh)
            
            # Step 2: Collect articles data
            articles_df = self.collect_articles_data(force_refresh=force_refresh)
            
            # Step 3: Score sentiment
            sentiment_df = self.score_articles_sentiment(articles_df, force_refresh=force_refresh)
            
            # Step 4: Combine datasets
            combined_df = self.create_combined_dataset(price_df, sentiment_df)
            
            # Step 5: Data quality summary
            self._print_data_summary(combined_df)
            
            print("=" * 60)
            print(f"[✅] Pipeline completed successfully!")
            print(f"[📊] Combined dataset saved to: {self.combined_cache_path}")
            
            return combined_df
            
        except Exception as e:
            print(f"[❌] Pipeline failed: {str(e)}")
            raise
    
    def _print_data_summary(self, df: pd.DataFrame):
        """Print a summary of the final dataset."""
        print("\n[📊] Dataset Summary:")
        print(f"  • Total records: {len(df)}")
        print(f"  • Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"  • Missing values: {df.isnull().sum().sum()}")


# Example usage and convenience functions
def create_ml_ready_dataset(ticker: str, company_name: str, api_key: str, 
                           start_date: str = "2015-01-01", 
                           cache_dir: str = "data",
                           force_refresh: bool = False) -> pd.DataFrame:
    """
    Convenience function to create a complete ML-ready dataset.
    
    Args:
        ticker (str): Stock ticker symbol
        company_name (str): Full company name
        api_key (str): NewsAPI key
        start_date (str): Start date for data collection
        cache_dir (str): Directory to store cached data
        force_refresh (bool): If True, refresh all cached data
        
    Returns:
        pandas.DataFrame: Complete dataset ready for ML
    """
    pipeline = StockAnalysisDataPipeline(
        ticker=ticker,
        company_name=company_name,
        api_key=api_key,
        cache_dir=cache_dir,
        start_date=start_date
    )
    
    return pipeline.run_full_pipeline(force_refresh=force_refresh)

if __name__ == "__main__":
    # Configuration
    TICKER = "NVDA"
    COMPANY_NAME = "NVIDIA"
    API_KEY = os.getenv("API_KEY")
    TARGET = "returns"
    
    # Create ML-ready dataset
    combined_df = create_ml_ready_dataset(
        ticker=TICKER,
        company_name=COMPANY_NAME,
        api_key=API_KEY,
        start_date="2024-01-01",
        force_refresh=False
    )