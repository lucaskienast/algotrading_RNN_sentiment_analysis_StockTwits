# Algotrading RNN sentiment analysis on StockTwits
This is a simple LSTM model written in Python to generate sentiment scores from posts on StockTwits. It works as follows:
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
