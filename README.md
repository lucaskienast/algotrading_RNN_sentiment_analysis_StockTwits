# Algotrading RNN sentiment analysis on StockTwits
In this project, a deep learning modelw as built to classify the sentiment of messages from StockTwits, a social network for investors and traders. The model will be able to predict if any particular message is positive or negative. From this, one can generate a signal of the public sentiment for various ticker symbols.

## Installation
Use `git clone` to get a copy of this repository.
```
$ git clone https://github.com/lucaskienast/algotrading_RNN_sentiment_analysis_StockTwits.git
$ cd algotrading_random_forest_enhanced_alpha
```

## Method
- get Twits JSON file split into messages and sentiments
- split messages from sentiments
- use `re` to remove URLs, tickers, usernames, non alphabetic characters, and `nltk.stem.WordNetLemmatizer` to lemmatize the rest
- create bag of words vocabulary to count up how often each word appears in the entire corpus using `collections.Counter`
- remove very common and rare words from vocabulary to reduce noise of input 
- balance sentiment classes to make sure each sentiment score shows up roughly as frequently as the others
- implement text classifier with `torch.nn`
- implement `dataloader` to loop through _StockTwits_ data whilst passing text sequences in as batches
- train model on GPU using `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- make prediction with random Twit example
- test model

## Results
It seems like the model is predicting the correct sentiment. The training loss fluctuated a lot between 0.7 and 0.8, whereas the validation loss appears to have remained relatively constant around 0.75 after the first epoch finished (same accounts for validation accuracy). Maybe there is some overfitting going on after the first epoch, so decreasing the number of epochs, or including early stopping or cross-validation would improve the accuracy of the model whilst minimizing overfitting.
