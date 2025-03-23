import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def setup_model():
    try:
        # Load FinBERT model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return tokenizer, model, device
    except Exception as e:
        raise Exception(f"Error loading the model: {str(e)}")

def get_sentiment(text, tokenizer, model, device):
    if pd.isna(text):
        return [0.333, 0.333, 0.333]  # Return equal probabilities for missing text
    
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities[0].cpu().numpy()
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error message: {str(e)}")
        return [0.333, 0.333, 0.333]

def main():
    try:
        # Load model and setup
        tokenizer, model, device = setup_model()
        
        # Read the news data
        df = pd.read_csv('AAPL_news_2019-2025_cleaned.csv')
        if df.empty:
            raise ValueError("The input CSV file is empty")
        
        # Process all titles with a progress bar
        print("Analyzing sentiment for all news titles...")
        sentiments = []
        for title in tqdm(df['title']):
            sentiment_scores = get_sentiment(title, tokenizer, model, device)
            sentiments.append(sentiment_scores)

        # Convert results to numpy array
        sentiments = np.array(sentiments)

        # Print sentiment scores for each title
        print("\nSentiment Analysis for Each Title:")
        print("-" * 100)
        for i, (title, scores) in enumerate(zip(df['title'], sentiments)):
            negative, neutral, positive = scores
            dominant_idx = np.argmax(scores)
            dominant = ['Negative', 'Neutral', 'Positive'][dominant_idx]
            print(f"\nTitle {i+1}: {title}")
            print(f"Positive: {positive:.3f}, Neutral: {neutral:.3f}, Negative: {negative:.3f}")
            print(f"Dominant Sentiment: {dominant}")

        # Export detailed results
        results_df = df.copy()
        results_df['sentiment_positive'] = sentiments[:, 2]
        results_df['sentiment_neutral'] = sentiments[:, 1]
        results_df['sentiment_negative'] = sentiments[:, 0]
        results_df['dominant_sentiment'] = ['Positive' if s == 2 else 'Neutral' if s == 1 else 'Negative' for s in np.argmax(sentiments, axis=1)]
        results_df.to_csv('sentiment_analysis_detailed.csv', index=False)

        print("\nResults exported to:")
        print("1. sentiment_analysis_detailed.csv - Contains sentiment scores for each article")

    except FileNotFoundError:
        print("Error: Could not find the input CSV file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
