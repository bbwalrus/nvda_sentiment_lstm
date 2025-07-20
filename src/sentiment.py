import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

def combine_text(row):
    """
    Helper function to combine title, description, and content into a single text string.
    If any part is missing, it will be skipped.
    """
    parts = [row.get("title", ""), row.get("description", ""), row.get("snippet", "")]
    return " ".join(part for part in parts if part)

def score_sentiment(csv_path: str, save=True, prob_scores=True):
    """
    Scores sentiment of articles in a CSV file using FinBERT.
    Args:
        csv_path (str): Path to the CSV file containing articles.
        save (bool): Whether to save the updated DataFrame back to the CSV file.
        prob_scores (bool): Whether to return probability scores for each sentiment class.
    Returns:
        pd.DataFrame: DataFrame with sentiment scores added.
    """
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} articles from {csv_path}")

    # Check if sentiment has already been scored
    sentiment_cols = ["positive", "neutral", "negative"]
    already_scored = all(col in df.columns for col in sentiment_cols)

    # Initialize sentiment columns if they don't exist
    if not already_scored:
        for col in sentiment_cols:
            df[col] = None
    
    # Only score rows where sentiment columns are missing or NaN
    if already_scored:
        mask = df[sentiment_cols].isnull().any(axis=1)
        rows_to_score = df[mask]
    else:
        mask = pd.Series([True] * len(df))  # Score all rows
        rows_to_score = df
    
    print(f"Found {len(rows_to_score)} rows to score")

    if rows_to_score.empty:
        print("All rows already have sentiment scores. No scoring needed.")
        return df  # Nothing to update

    # Combine title + description + content for sentiment input
    texts = rows_to_score.apply(combine_text, axis=1).tolist()
    
    # Filter out empty texts
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text.strip():  # Only process non-empty texts
            valid_texts.append(text)
            valid_indices.append(rows_to_score.index[i])
    
    if not valid_texts:
        print("No valid texts found to score.")
        return df

    print(f"Scoring sentiment for {len(valid_texts)} valid texts...")

    # Load FinBERT model & pipeline
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    model_pipeline = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        return_all_scores=prob_scores
    )

    # Run sentiment scoring in batches to avoid memory issues
    batch_size = 32
    all_sentiment_results = []
    
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i+batch_size]
        batch_results = model_pipeline(batch_texts, truncation=True, max_length=512)
        all_sentiment_results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}/{(len(valid_texts) + batch_size - 1)//batch_size}")

    # Convert sentiment scores into structured format
    for idx, (original_idx, article_scores) in enumerate(zip(valid_indices, all_sentiment_results)):
        for entry in article_scores:
            label = entry["label"].lower()
            score = entry["score"]
            df.at[original_idx, label] = score
    
    print(f"Successfully scored sentiment for {len(valid_indices)} articles")
    
    # Verify that sentiment columns were added
    missing_sentiment = df[sentiment_cols].isnull().sum()
    print(f"Missing sentiment scores after processing: {missing_sentiment.to_dict()}")

    # Save updated CSV
    if save:
        df.to_csv(csv_path, index=False)
        print(f"Saved updated sentiment scores to: {csv_path}")
        
        # Verify the saved file has the columns
        verification_df = pd.read_csv(csv_path)
        if all(col in verification_df.columns for col in sentiment_cols):
            print("✓ Verified: Sentiment columns saved successfully")
        else:
            print("✗ Warning: Sentiment columns not found in saved file")

    return df

# Example usage for debugging
if __name__ == "__main__":
    # Example usage
    csv_path = "data/NVDA_articles.csv"
    
    # Debug the sentiment scoring
    #debug_sentiment_scoring(csv_path)
    
    # Or just run the fixed version
    result_df = score_sentiment(csv_path, save=True, prob_scores=True)
    
    # Or check and repair
    # result_df = check_and_repair_sentiment(csv_path)