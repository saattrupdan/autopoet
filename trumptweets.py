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

    def __init__(self, workers = None):
        import os
        import pandas as pd
        import multiprocessing as mp

        if workers is None:
            self.workers = mp.cpu_count()
        else:
            self.workers = min(mp.cpu_count(), workers)

        if os.path.isfile('data/sents.tsv'):
            self.sents = pd.read_csv('data/sents.tsv', sep = '\t', 
                dtype = {'sentences': str, 'syllables': int})
        else:
            self.sents = None

        if os.path.isfile('data/vocab.tsv'):
            self.vocab = pd.read_csv('data/vocab.tsv', sep = '\t', 
                dtype = {'words': str, 'syllables': int})
        else:
            self.vocab = None

    def map_docs(self, docs, fn, pbar_desc = 'Parsing documents'):
        ''' Apply function to iterable of documents. '''
        import multiprocessing as mp
        from tqdm import tqdm
        if self.workers == 1:
            with tqdm(docs) as pbar:
                pbar.set_description(pbar_desc)
                return list(map(fn, pbar))
        else: 
            with mp.Pool(self.workers) as pool:
                with tqdm(docs) as pbar:
                    pbar.set_description(pbar_desc)
                    return list(pool.map(fn, pbar))
        
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
        import pandas as pd

        unpacked_tweets = self.unpack_docs(tweets)

        # Load SpaCy's English NLP model to tokenize and add emoji support
        nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
        nlp.add_pipe(Emoji(nlp, merge_spans = False), first = True)

        # Parse clean tweets
        unpacked_tweets = self.map_docs(unpacked_tweets, fn = nlp,
            pbar_desc = 'Adding parts-of-speech tags to words')

        # Build vocabulary 
        self.vocab = list(set([word.text 
            for tweet in unpacked_tweets for word in tweet
            if word.pos_ not in {'PUNCT', 'SYM', 'SPACE', 'X'}
            and not word._.is_emoji
            ]))

        # Count syllables in vocab and store in dataframe
        counter = SyllableCounter()
        self.vocab = pd.DataFrame({
            'word': self.vocab, 
            'syllables': counter.count_docs(self.vocab)
            })

        # Remove both blank sentences and sentences that led the syllable
        # count to encounter an error
        self.vocab.dropna(inplace = True)
        self.vocab = self.vocab[self.vocab['syllables'] > 0]

        # Save vocab
        self.vocab.to_csv('data/vocab.tsv', index = False, sep = '\t')

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

        # Load SpaCy's English NLP model to parse tweets
        nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'ner'])

        # Parse tweets
        tweets = self.map_docs(tweets, fn = nlp,
            pbar_desc = 'Splitting tweets into sentences')

        # Extract sentences from tweets and clean them
        self.sents = [sent.text for tweet in tweets for sent in tweet.sents]

        # Load SpaCy's English NLP model to parse sentences
        nlp = spacy.load('en_core_web_sm', 
            disable = ['tagger', 'parser', 'ner'])

        # Parse sentences
        unpacked_sents = self.map_docs(self.unpack_docs(self.sents), fn = nlp,
            pbar_desc = 'Splitting sentences into words')

        # Count sentence syllables
        syllables = self.map_docs(unpacked_sents, fn = self.sent2syls,
            pbar_desc = 'Counting sentence syllables')

        self.sents = pd.DataFrame({
            'sentence': self.clean_docs(self.sents), 
            'syllables': syllables
            })

        # Remove blank sentences
        self.sents.dropna(inplace = True)
        self.sents = self.sents[self.sents['syllables'] > 0]

        # Save sentences
        self.sents.to_csv('data/sents.tsv', index = False, sep = '\t')

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

        if self.vocab is None:
            self.build_vocab(tweets)

        if self.sents is None:
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

            # Remove punctuation at the beginning or end of phrase
            phrase = re.sub(r'([.,\-… \'\"\”]+$|^[\'\"\”.,\-… ]+)', '', phrase)

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

    tt = TrumpTweets(workers = 1)
    tt.compile()

    print('HAIKU 1:')
    print(tt.haiku())

    print('\nHAIKU 2:')
    print(tt.haiku())

    print('\nHAIKU 3:')
    print(tt.haiku())
