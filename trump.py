class SyllableCounter():
    ''' Count number of syllables in text. '''

    def __init__(self):
        from big_phoney import BigPhoney
        self.phoney = BigPhoney()

    def count_doc(self, doc):
        ''' Count syllables of a string.

        INPUT
            doc
                An string
        OUTPUT
            A syllable count
        '''
        import numpy as np
        if doc == '':
            return 0
        else:
            try:
                return self.phoney.count_syllables(doc)
            except:
                return np.nan

    def count_docs(self, docs):
        ''' Count syllables in each string in input iterable.

        INPUT
            docs
                An iterable of strings
        OUTPUT
            A list of syllable counts.
        '''
        from tqdm import tqdm
        with tqdm(docs) as pbar:
            pbar.set_description('Counting syllables')
            return list(map(self.count_doc, pbar))

class TrumpTweets():
    ''' Trump tweets from http://www.trumptwitterarchive.com/archive. '''

    def __init__(self):
        self.tweets = None

    def compile(self, json_file = 'tweets.json'):
        ''' Load in the tweets from a json file, split into phrases, count
            syllables and save as a tsv file to be loaded by load().
    
            INPUT
                fname_in = 'tweets.json'
                    The json file containing all the tweets
                fname_out = 'tweets.tsv'
                    The tsv file in which the dataframe will be stored
        '''
        import json
        import spacy
        from tqdm import tqdm
        import re
        import pandas as pd

        with open(json_file, 'r') as file_in:
            tweets = json.load(file_in)
            tweets = [tweet['text'] for tweet in tweets]

        # Load SpaCy's English NLP model 
        nlp = spacy.load('en_core_web_sm', disable = ['ner'])
       
        # Clean tweets
        with tqdm(tweets) as pbar:
            pbar.set_description('Cleaning tweets')
            re_link = re.compile(r'http[./:a-zA-Z0-9]+')
            re_number = re.compile(r'[^ ]*[0-9][^ ]*')
            re_spaces = re.compile(r' +')
            re_dash = re.compile(r'-+')
            clean_tweets = [re.sub(re_spaces, ' ', re.sub(re_link, ' ',
                re.sub(re_number, ' ', re.sub(re_dash, ' ', 
                re.sub('@', 'at-', tweet))))).lower()
                for tweet in pbar]

        # Parse tweets
        with tqdm(clean_tweets) as pbar:
            pbar.set_description('Parsing tweets')
            clean_tweets = [nlp(tweet) for tweet in pbar]
       
        # Get vocabulary 
        vocab = list(set([word.text 
            for tweet in clean_tweets for word in tweet
            if word.pos_ not in {'PUNCT', 'SYM', 'SPACE', 'X', 'NUM'}]))

        # Count syllables in vocab and store in dataframe
        counter = SyllableCounter()
        vocab = {'word': vocab, 'syllables': counter.count_docs(vocab)}
        vocab = pd.DataFrame(vocab)

        # Remove both blank sentences and sentences that led the syllable
        # count to encounter an error
        vocab = vocab[vocab['count'] > 0]
        vocab.dropna(inplace = True)

        # Save vocab
        vocab.to_csv('vocab.tsv', index = False, sep = '\t')
        self.vocab = vocab

        # Use syllable counts of words to count syllables in tweets
        def count_syllables(clean_tweets):
            syllables = [sum(vocab.loc[vocab['word'] == word.text]\
                ['syllables'] for word in clean_tweet) 
                for clean_tweet in clean_tweets]
            return syllables
        tweets = {'tweet': tweets, 'syllables': count_syllables(clean_tweets)}
        tweets = pd.DataFrame(tweets)

        # Save tweets
        tweets.to_csv('tweets.tsv', index = False, sep = '\t')
        self.tweets = tweets

        return self

    def load(self, fname = 'tweets.tsv'):
        ''' Load compiled tweets from a tsv file.

        INPUT
            fname = 'tweets.tsv'
                The file name to be loaded
        '''
        import pandas as pd
        self.tweets = pd.read_csv(fname, sep = '\t')
        return self

    def rnd_phrase(self, syllables = None):
        ''' Get a random Trump phrase.

        INPUT
            syllables = None
                The number of syllables in phrase. Defaults to no syllable
                requirement
        OUTPUT
            A random Trump phrase, with first letter capitalised
        '''
        import random
        import re

        if syllables:
            phrases = self.tweets[self.tweets['count'] == syllables]['tweet']
        else:
            phrases = self.tweets['tweet']

        try:
            phrases = list(phrases)
            phrase = random.choice(list(phrases))

            # Make the first letter in the phrase uppercase
            phrase = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), phrase)

        except IndexError:
            phrase = f'<No phrase found with {syllables} syllables>'

        return phrase

    def haiku(self):
        ''' Get a random Trump haiku. 
        
        OUTPUT
            Trump haiku string
        '''
        line1 = self.rnd_phrase(5)
        line2 = self.rnd_phrase(7)
        line3 = self.rnd_phrase(5)
        return line1 + '\n' + line2 + '\n' + line3

if __name__ == '__main__':

    tt = TrumpTweets()
    tt.compile()
    #tt.load()

    #print('\nHAIKU 1:')
    #print(tt.haiku())

    #print('\nHAIKU 2:')
    #print(tt.haiku())

    #print('\nHAIKU 3:')
    #print(tt.haiku())
