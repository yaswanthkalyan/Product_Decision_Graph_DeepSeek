# Product Purchase Decision Maker 🛒

## Overview
The **Product Purchase Decision Maker** is a Streamlit-based web application that helps users analyze a product's pros and cons and decide whether to buy or skip it. The app leverages **Groq AI models** to evaluate product reviews, extract relevant information, and provide a balanced decision.

## Features
✅ **Sentiment Analysis** – Uses AI to assess positive and negative arguments.
✅ **Real-time Web Scraping** – Extracts product details from the provided URL.
✅ **Balanced Decision Making** – Ensures both pros and cons are fairly weighed before making a recommendation.
✅ **User-Friendly Interface** – Simple and intuitive Streamlit-based UI.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- pip
- An OpenAI-compatible Groq API key
- Streamlit
- Pydantic AI, Pydantic Graph, and Tavily API client

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/product-decision-maker.git
   cd product-decision-maker
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   Create a `.env` file and add:
   ```
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the browser at `http://localhost:8501`.
3. Enter the **product URL** and relevant **keywords**.
4. Click **"Analyze Product"** to get the decision.

## Decision-Making Process
1. **Pro-Agent Analysis** – Generates positive arguments with sentiment scores (0 to 100).
2. **Con-Agent Analysis** – Generates negative arguments with sentiment scores (-100 to 0).
3. **Sentiment Balancing** – Weighs both sides to ensure fairness.
4. **Final Decision**
   - **Buy** if positive sentiment outweighs negative by **at least 15 points**.
   - **Skip** if negative sentiment is **15 points higher**.
   - **Neutral/Skip** if both arguments are close.

## Example Output
```
✅ Decision: Buy
💡 Reasoning: The product received highly positive feedback, with durable quality and great user reviews. The positive sentiment significantly outweighed the negatives.
```

## Future Improvements
- Support for more **customizable decision parameters**.
- Expand product sources beyond single URL scraping.
- Enhance **argument quality** by incorporating more detailed web searches.

## License
This project is licensed under the MIT License. Feel free to contribute!

