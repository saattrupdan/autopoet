class SyllableCounter():
    ''' Count number of syllables in text. '''

    def __init__(self):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Suppress Tensorflow warnings
            from tensorflow.python.util import deprecation
            import os
            deprecation._PRINT_DEPRECATION_WARNINGS = False
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            from big_phoney import BigPhoney
            self.phoney = BigPhoney()

    def count_doc(self, doc):
        ''' Count syllables of a document.

        INPUT
            doc
                A document
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
        import os
        import pandas as pd

        if os.path.isfile('sents.tsv'):
            self.sents = pd.read_csv('sents.tsv', sep = '\t')
        else:
            self.sents = None

        if os.path.isfile('vocab.tsv'):
            self.vocab = pd.read_csv('vocab.tsv', sep = '\t')
        else:
            self.vocab = None

    def clean_docs(self, docs):
        ''' Clean an iterable of docs. '''
        import re

        re_link = re.compile(r'http[./:a-zA-Z0-9]+')
        re_spaces = re.compile(r' +')
        re_dash = re.compile(r'-+')

        clean_docs = [
            re.sub(re_spaces, ' ', 
            re.sub(re_link, ' ',
            re.sub(re_dash, ' ', doc))) 
            for doc in docs
            ]

        return clean_docs

    def unpack_docs(self, docs):
        ''' Convert docs into a format that allows syllable counting. '''
        import re
        clean_docs = self.clean_docs(docs)
        return [re.sub('@', 'at-', doc).lower() for doc in clean_docs]

    def build_vocab(self, tweets):
        ''' Build vocabulary, with syllable counts. '''
        import spacy
        from spacymoji import Emoji
        from tqdm import tqdm
        import pandas as pd
        import multiprocessing as mp

        unpacked_tweets = self.unpack_docs(tweets)

        # Load SpaCy's English NLP model to tokenize and add emoji support
        nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
        nlp.add_pipe(Emoji(nlp, merge_spans = False), first = True)

        # Parse clean tweets
        with mp.Pool() as pool:
            with tqdm(unpacked_tweets) as pbar:
                pbar.set_description('Adding parts-of-speech tags to words')
                unpacked_tweets = list(pool.map(nlp, pbar))
            
        # Build vocabulary 
        vocab = list(set([word.text 
            for tweet in unpacked_tweets for word in tweet
            if word.pos_ not in {'PUNCT', 'SYM', 'SPACE', 'X'}
            and not word._.is_emoji
            ]))

        # Count syllables in vocab and store in dataframe
        counter = SyllableCounter()
        vocab = {'word': vocab, 'syllables': counter.count_docs(vocab)}
        vocab = pd.DataFrame(vocab)

        # Remove both blank sentences and sentences that led the syllable
        # count to encounter an error
        vocab = vocab[vocab['syllables'] > 0]
        vocab.dropna(inplace = True)

        # Save vocab
        vocab.to_csv('vocab.tsv', index = False, sep = '\t')
        self.vocab = vocab

        return self

    def word2syls(self, word):
        ''' Count syllables in word. '''
        df = self.vocab.loc[self.vocab['word'] == word.text]['syllables']
        if df.size > 0:
            return list(df)[0]
        else:
            return 0

    def sent2syls(self, sent):
        ''' Count syllables in sentence. '''
        return sum(self.word2syls(word) for word in sent)

    def build_sents(self, tweets):
        ''' Extract sentences from tweets, with syllable counts. '''
        from tqdm import tqdm
        import pandas as pd
        import spacy
        import multiprocessing as mp

        # Load SpaCy's English NLP model to parse sentences
        nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'ner'])

        # Parse tweets
        with mp.Pool() as pool:
            with tqdm(tweets) as pbar:
                pbar.set_description('Splitting tweets into sentences')
                tweets = list(pool.map(nlp, pbar))
            pool.close()
            pool.join()

        # Extract sentences from tweets and clean them
        sents = [sent.text for tweet in tweets for sent in tweet.sents]
        clean_sents = self.clean_docs(sents)
        unpacked_sents = self.unpack_docs(sents)

        # Parse sentences
        with mp.Pool() as pool:
            with tqdm(unpacked_sents) as pbar:
                pbar.set_description('Splitting sentences into words')
                unpacked_sents = list(pool.map(nlp, pbar))
            pool.close()
            pool.join()

        # Count sentence syllables
        with mp.Pool() as pool:
            with tqdm(unpacked_sents) as pbar:
                pbar.set_description('Counting sentence syllables')
                syllables = list(pool.map(self.sent2syls, pbar))
                sents = {'sentence': clean_sents, 'syllables': syllables}
                sents = pd.DataFrame(sents)
            pool.close()
            pool.join()

        # Remove blank sentences
        sents = sents[sents['syllables'] > 0]
        sents.dropna(inplace = True)

        # Save sentences
        sents.to_csv('sents.tsv', index = False, sep = '\t')
        self.sents = sents

        return self

    def compile(self, json_fname = 'tweets.json'):
        ''' Load in the tweets from a json file, build vocabulary, split
            into sentences and count syllables.
    
            INPUT
                json_fname = 'tweets.json'
                    The json file containing all the tweets
        '''
        import json
        import os
        import pandas as pd

        with open(json_fname, 'r') as file_in:
            tweets = json.load(file_in)
            tweets = [tweet['text'] for tweet in tweets]

        if os.path.isfile('vocab.tsv'):
            self.vocab = pd.read_csv('vocab.tsv', sep = '\t')
        else:
            self.build_vocab(tweets)

        if os.path.isfile('sents.tsv'):
            self.sents = pd.read_csv('sents.tsv', sep = '\t')
        else:
            self.build_sents(tweets)

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

        while self.sents is None:
            query = 'You have not compiled the tweets yet. '\
                'Compile? (y/n)\n >>> '
            if input(query) == 'y':
                self.compile()

        if syllables:
            phrases = self.sents[self.sents['syllables'] == syllables]
            phrases = phrases['sentence']
        else:
            phrases = self.sents['sentence']

        try:
            phrases = list(phrases)
            phrase = random.choice(list(phrases))

            # Make the first letter in the phrase uppercase
            phrase = re.sub(r'^[a-z]', lambda m: m.group(0).upper(), phrase)

        except IndexError:
            phrase = f'<No phrase found with {syllables} syllables>'

        return phrase.strip()

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
