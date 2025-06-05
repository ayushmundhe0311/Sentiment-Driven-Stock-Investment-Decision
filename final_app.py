import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from final_main import run_analysis, train_lstm, predict_next_price, scrape_stock_news, query_sentiment
import time

# Streamlit app configuration
st.set_page_config(
    page_title="Stock Prediction & Recommendation",
    layout="wide",
    page_icon="📈"
)

def display_sentiment(scores):
    if not scores:
        st.warning("No sentiment scores available")
        return
        
    if isinstance(scores, list) and len(scores) > 0 and isinstance(scores[0], list):
        scores = scores[0]
        
    sentiment_df = pd.DataFrame(scores)
    st.dataframe(
        sentiment_df.style.highlight_max(axis=0, color='lightgreen'),
        use_container_width=True
    )

def basic_recommendation(current_price, predicted_price):
    price_diff = ((predicted_price - current_price)/current_price)*100
    if price_diff > 2.5:
        return "📈 Basic Recommendation: POTENTIAL BUY - Significant predicted price increase"
    elif price_diff < -2.5:
        return "📉 Basic Recommendation: POTENTIAL SELL - Significant predicted price decrease"
    else:
        return "🤝 Basic Recommendation: HOLD - Moderate price movement predicted"

def show_progress_step(step, total_steps, message):
    progress = step / total_steps
    with st.status(f"Step {step}/{total_steps}: {message}", expanded=True) as status:
        progress_bar = st.progress(progress)
        time.sleep(0.5)  # For better UX
        return status

def main():
    st.title("📈 Stock Prediction & Recommendation System")
    
    # Available stocks (extracted from your data columns)
    available_stocks = [
        "BHARTIARTL_NS", "HDFCBANK_NS", "HINDUNILVR_NS", 
        "ICICIBANK_NS", "INFY_NS", "LT_NS", "RELIANCE_NS", 
        "SBIN_NS", "TATAMOTORS_NS", "TCS_NS"
    ]
    
    # User inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock = st.selectbox("Select Stock", available_stocks)
    
    with col2:
        target_date = st.date_input(
            "Select Target Date",
            min_value=datetime.today() + timedelta(days=0),
            max_value=datetime.today() + timedelta(days=365),
            value=datetime.today() + timedelta(days=7)
        )
    
    with col3:
        current_price = st.number_input(
            f"Current Price of {stock.split('_')[0]}",
            min_value=0.01,
            value=100.0,
            step=0.1,
            format="%.2f"
        )
    
    if st.button("Get Prediction & Recommendation", type="primary"):
        with st.spinner("Initializing analysis..."):
            try:
                date_str = target_date.strftime("%Y-%m-%d")
                stock_name = stock.replace("_NS", "")
                
                # Step 1: Price Prediction
                with show_progress_step(1, 3, "Running price prediction...") as status:
                    try:
                        # Check if model exists, otherwise train
                        if not st.session_state.get(f"{stock}_model_trained", False):
                            train_lstm(stock)
                            st.session_state[f"{stock}_model_trained"] = True
                        
                        predicted_price = predict_next_price(stock, date_str)
                        status.update(label="Price prediction complete!", state="complete")
                    except Exception as e:
                        status.update(label=f"Price prediction failed: {str(e)}", state="error")
                        st.error(f"Failed to predict price: {str(e)}")
                        return
                
                # Display prediction results
                st.subheader("📊 Prediction Results")
                col_pred1, col_pred2 = st.columns(2)
                col_pred1.metric(
                    "Current Price", 
                    f"₹{current_price:.2f}",
                    help="The current market price you entered"
                )
                
                price_diff = ((predicted_price - current_price)/current_price)*100
                col_pred2.metric(
                    "Predicted Price", 
                    f"₹{predicted_price:.2f}", 
                    delta=f"{price_diff:.2f}%",
                    help=f"Predicted price for {target_date.strftime('%b %d, %Y')}"
                )
                
                # Step 2: News Collection
                with show_progress_step(2, 3, "Gathering recent news...") as status:
                    try:
                        news_text = scrape_stock_news(stock_name, target_date)
                        if not news_text:
                            status.update(label="No recent news found", state="complete")
                            st.warning("No recent news found. Showing prediction without sentiment analysis.")
                            st.subheader("✅ Investment Recommendation")
                            st.info(basic_recommendation(current_price, predicted_price))
                            return
                        status.update(label="News collection complete!", state="complete")
                    except Exception as e:
                        status.update(label=f"News collection failed: {str(e)}", state="error")
                        st.warning(f"News collection failed: {str(e)}. Showing basic recommendation.")
                        st.subheader("✅ Investment Recommendation")
                        st.info(basic_recommendation(current_price, predicted_price))
                        return
                
                # Step 3: Sentiment Analysis
                with show_progress_step(3, 3, "Analyzing news sentiment...") as status:
                    try:
                        sentiment_scores = query_sentiment(news_text)
                        if not sentiment_scores:
                            status.update(label="Sentiment analysis failed", state="complete")
                            st.warning("Sentiment analysis failed. Showing basic recommendation.")
                            st.subheader("✅ Investment Recommendation")
                            st.info(basic_recommendation(current_price, predicted_price))
                            return
                        status.update(label="Sentiment analysis complete!", state="complete")
                    except Exception as e:
                        status.update(label=f"Sentiment analysis failed: {str(e)}", state="error")
                        st.warning(f"Sentiment analysis failed: {str(e)}. Showing basic recommendation.")
                        st.subheader("✅ Investment Recommendation")
                        st.info(basic_recommendation(current_price, predicted_price))
                        return
                
                # Display full analysis results
                st.subheader("📰 Latest News Analysis")
                with st.expander("View News Articles"):
                    st.text(news_text)
                
                st.write("### Sentiment Analysis")
                display_sentiment(sentiment_scores)
                
                # Generate final recommendation
                st.subheader("✅ Investment Recommendation")
                recommendation = rule_based_recommendation(current_price, predicted_price, sentiment_scores)
                if "BUY" in recommendation:
                    st.success(recommendation)
                elif "SELL" in recommendation:
                    st.error(recommendation)
                else:
                    st.info(recommendation)
                    
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.exception(e)

def rule_based_recommendation(current_price, predicted_price, sentiment_scores):
    if sentiment_scores is None:
        return basic_recommendation(current_price, predicted_price)
        
    if isinstance(sentiment_scores, list) and len(sentiment_scores) > 0 and isinstance(sentiment_scores[0], list):
        sentiment_scores = sentiment_scores[0]

    pos = next((s['score'] for s in sentiment_scores if s['label'] == 'positive'), 0)
    neg = next((s['score'] for s in sentiment_scores if s['label'] == 'negative'), 0)
    neu = next((s['score'] for s in sentiment_scores if s['label'] == 'neutral'), 0)

    price_diff_percent = ((predicted_price - current_price) / current_price) * 100
    
    st.write(f"**Price Change:** {price_diff_percent:.2f}%")
    st.write(f"**Sentiment Scores:** Positive: {pos:.2f}, Negative: {neg:.2f}, Neutral: {neu:.2f}")

    if price_diff_percent > 2.5 and pos > 0.5 and pos > (neg + neu):
        return "📈 Recommendation: STRONG BUY - Significant upside potential with positive market sentiment"
    elif price_diff_percent > 1.5 and pos > 0.4:
        return "📈 Recommendation: BUY - Positive price movement with favorable sentiment"
    elif price_diff_percent < -2.5 and (neg > pos and neg > 0.4):
        return "📉 Recommendation: STRONG SELL - Significant downside risk with negative sentiment"
    elif price_diff_percent < -1.5 and neg > 0.4:
        return "📉 Recommendation: SELL - Negative price movement with unfavorable sentiment"
    elif abs(price_diff_percent) < 1.0 and neu > 0.6:
        return "🔄 Recommendation: HOLD - Stable price with neutral sentiment"
    else:
        return "🤝 Recommendation: HOLD - Market shows mixed signals"

if __name__ == "__main__":
    main()