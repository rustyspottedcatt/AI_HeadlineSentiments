# SmartSentiment: Self-Learning Financial News Sentiment Analyzer

SmartSentiment is a dynamic C# sentiment analysis application designed to predict the sentiment of financial news headlines, identifying positive and negative tones in market-related news. Built using ML.NET, it provides users with real-time sentiment predictions and evolves to improve its accuracy with continuous feedback and live data.

## Features
- **Sentiment Prediction**: Analyzes and predicts sentiment for financial news headlines.
- **Self-Learning**: Continuously improves by learning from user feedback and integrating live news headlines from the News API.
- **User Feedback**: Allows users to correct predictions, which adds new labeled data for retraining the model.
- **Periodic Model Retraining**: Updates its model over time to stay relevant and accurate.

## Getting Started
1. Clone the repository and set up the necessary dependencies.
2. Sign up for an API key on [News API](https://newsapi.org/) to fetch live news headlines.
3. Run the application and input financial news headlines to get sentiment predictions or fetch new headlines for analysis.

## Requirements
- .NET SDK
- ML.NET
- API key from News API for live news integration

## Usage
```shell
dotnet run
