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
            A syllable count.
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
        pbar = tqdm(iter(docs), total = len(docs))
        pbar.set_description('Counting syllables')
        return list(map(self.count_doc, pbar))

class TrumpTweets():
    ''' Trump tweets from http://www.trumptwitterarchive.com/archive. '''

    def __init__(self):
        self.tweets = None

    def compile(self, fname_in = 'tweets.json', fname_out = 'tweets.csv'):
        ''' Load in the tweets from a json file, split into phrases, count
            syllables and save as a Pandas dataframe.
    
            INPUT
                fname_in = 'tweets.json'
                    The json file containing all the tweets
                fname_out = 'tweets.csv'
                    The csv file in which the dataframe will be stored
        '''
        import json
        import re
        from itertools import chain
        import pandas as pd

        with open(fname_in, 'r') as file_in:
            tweets = json.load(file_in)

        tweets = (re.split(r' *[.,!]($| +)', 
                  re.sub(r'([0-9]),([0-9])', r'\1\2', 
                  re.sub(r'^\.+', '', 
                  re.sub(r' +', ' ', tweet['text']))))
                  for tweet in tweets)
        tweets = list(chain.from_iterable(tweets))

        tweets = tweets

        counter = SyllableCounter()
        tweets = {'tweet': tweets, 'count': counter.count_docs(tweets)}
        tweets = pd.DataFrame(tweets)
        tweets = tweets[tweets['count'] > 0]
        tweets.dropna(inplace = True)
        tweets.to_csv(fname_out, index = False)

        self.tweets = tweets

        return self

    def load(self, fname = 'tweets.csv'):
        ''' Load compiled tweets from a csv file.

        INPUT
            fname = 'tweets.csv'
                The file name to be loaded
        '''
        self.tweets = pd.read_csv(fname_in)
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

    print('\nHAIKU 1:')
    print(tt.haiku())

    print('\nHAIKU 2:')
    print(tt.haiku())

    print('\nHAIKU 3:')
    print(tt.haiku())
