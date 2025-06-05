import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import joblib
import requests
from datetime import datetime, timedelta
from googlesearch import search
from newspaper import Article
import textwrap

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
headers = {
    "Authorization": "Bearer API"  # Replace with your actual key
}

def train_lstm(stock_name):
    close_col = f"{stock_name}_Close"
    df = pd.read_csv("data.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    if close_col not in df.columns:
        raise ValueError(f"❌ Stock column '{close_col}' not found in dataset.")

    data = df[[close_col]].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{stock_name}_lstm_model.h5")
    joblib.dump(scaler, f"models/{stock_name}_scaler.save")

def predict_next_price(stock_name, target_date_str):
    close_col = f"{stock_name}_Close"
    df = pd.read_csv("data.csv", parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    model_path = f"models/{stock_name}_lstm_model.h5"
    scaler_path = f"models/{stock_name}_scaler.save"
    
    if not os.path.exists(model_path):
        train_lstm(stock_name)
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    data = df[["Date", close_col]].dropna()
    past_data = data[data["Date"] < pd.to_datetime(target_date_str)]

    if len(past_data) == 0:
        raise ValueError("❌ No historical data available before target date.")

    last_values = past_data.tail(60)[close_col].values.tolist()

    while len(last_values) < 60:
        input_seq = np.array(last_values).reshape(-1, 1)
        input_seq_scaled = scaler.transform(input_seq)
        padded = np.pad(input_seq_scaled, ((60 - len(input_seq), 0), (0, 0)), 'edge')
        prediction = model.predict(padded.reshape(1, 60, 1), verbose=0)
        predicted_value = scaler.inverse_transform(prediction)[0][0]
        last_values.append(predicted_value)

    final_input = np.array(last_values[-60:]).reshape(-1, 1)
    final_input_scaled = scaler.transform(final_input)
    predicted_price = model.predict(final_input_scaled.reshape(1, 60, 1), verbose=0)
    final_price = scaler.inverse_transform(predicted_price)[0][0]

    return final_price

def scrape_stock_news(stock_name, target_date, max_words=350):
    print(f"\n🔍 Scraping news for: {stock_name} between {target_date - timedelta(days=2)} and {target_date}")
    query = f"{stock_name} stock news {target_date.strftime('%Y-%m-%d')}"
    
    try:
        # Try multiple search queries if needed
        queries = [
            f"{stock_name} stock news {target_date.strftime('%Y-%m-%d')}",
            f"{stock_name} share price news",
            f"{stock_name} company news"
        ]
        
        articles = []
        word_count = 0
        combined_text = ""
        
        for q in queries:
            try:
                print(f"Trying search query: {q}")
                results = search(q, num_results=10, sleep_interval=2)
                
                if not results:
                    continue
                    
                for url in results:
                    try:
                        if not url:
                            continue
                            
                        # Skip unwanted domains
                        skip_domains = ["youtube", "login", "nseindia", "bseindia", "moneycontrol", "twitter"]
                        if any(skip in url.lower() for skip in skip_domains):
                            continue
                            
                        print(f"Processing URL: {url}")
                        article = Article(url, request_timeout=10)
                        article.download()
                        article.parse()
                        
                        if not article.text or len(article.text) < 100:
                            continue
                            
                        title = article.title or "No title"
                        text = article.text
                        summary = textwrap.fill(text.strip(), width=100, max_lines=4, placeholder="...")
                        
                        total_words = len(title.split()) + len(summary.split())
                        if word_count + total_words > max_words:
                            break

                        articles.append({"title": title, "summary": summary})
                        word_count += total_words
                        combined_text += f"{title}\n{summary}\n\n"
                        
                        if len(articles) >= 5:
                            break
                            
                    except Exception as e:
                        print(f"⚠️ Failed to process article: {str(e)}")
                        continue
                        
                if articles:
                    break
                    
            except Exception as e:
                print(f"⚠️ Search failed for query '{q}': {str(e)}")
                continue

        if not articles:
            print("❌ No valid news articles found.")
            return ""
        
        print(f"Found {len(articles)} articles")
        for i, a in enumerate(articles, start=1):
            print(f"{i}. 📰 {a['title']}\n   {a['summary']}\n")

        return combined_text.strip()
        
    except Exception as e:
        print(f"❌ News scraping failed completely: {str(e)}")
        return ""
    
def query_sentiment(news_text):
    if not news_text.strip():
        return None
        
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": news_text}, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {str(e)}")
        return None

def rule_based_recommendation(current_price, predicted_price, sentiment_scores):
    if sentiment_scores is None:
        return "⚠️ No sentiment analysis available"
        
    if isinstance(sentiment_scores, list) and len(sentiment_scores) > 0 and isinstance(sentiment_scores[0], list):
        sentiment_scores = sentiment_scores[0]

    pos = next((s['score'] for s in sentiment_scores if s['label'] == 'positive'), 0)
    neg = next((s['score'] for s in sentiment_scores if s['label'] == 'negative'), 0)
    neu = next((s['score'] for s in sentiment_scores if s['label'] == 'neutral'), 0)

    price_diff_percent = ((predicted_price - current_price) / current_price) * 100
    print(f"\n📉 Price Change: {price_diff_percent:.2f}%")
    print(f"💬 Sentiment - Positive: {pos:.2f}, Negative: {neg:.2f}, Neutral: {neu:.2f}")

    if price_diff_percent > 2.5 and pos > 0.5 and pos > (neg + neu):
        return "📈 Recommendation: BUY - Strong upside potential with positive sentiment."
    elif price_diff_percent < -2.5 and (neg > pos and neg > 0.4):
        return "📉 Recommendation: SELL - Downtrend with negative sentiment."
    else:
        return "🤝 Recommendation: HOLD - Market uncertain or low price movement."

def run_analysis(stock, target_date_str, current_price):
    try:
        close_col = f"{stock}_Close"
        df = pd.read_csv("data.csv", parse_dates=["Date"])
        df.sort_values("Date", inplace=True)

        if close_col not in df.columns:
            return None, None, None, f"❌ Stock column '{close_col}' not found in dataset."

        if not os.path.exists(f"models/{stock}_lstm_model.h5"):
            train_lstm(stock)

        predicted_price = predict_next_price(stock, target_date_str)
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        news_text = scrape_stock_news(stock.replace("_NS", ""), target_date)
        
        if news_text:
            sentiment_scores = query_sentiment(news_text)
            if sentiment_scores:
                recommendation = rule_based_recommendation(current_price, predicted_price, sentiment_scores)
                return predicted_price, news_text, sentiment_scores, recommendation
            return predicted_price, news_text, None, "⚠️ Sentiment analysis failed"
        return predicted_price, None, None, "⚠️ No news found for sentiment analysis"
            
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"