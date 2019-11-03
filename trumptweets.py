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

        if os.path.isfile(os.path.join('data', 'sents.tsv')):
            self.sents = pd.read_csv(os.path.join('data', 'sents.tsv'), 
                sep = '\t', dtype = {'sentences': str, 'syllables': int})
        else:
            self.sents = None

    def map_docs(self, docs, fn, pbar_desc = 'Parsing documents',
        workers = None):
        ''' Apply function to iterable of documents. '''
        import multiprocessing as mp
        from tqdm import tqdm

        if workers is None:
            workers = self.workers

        if workers == 1:
            with tqdm(docs) as pbar:
                pbar.set_description(pbar_desc)
                return list(map(fn, pbar))
        else: 
            with mp.Pool(self.workers) as pool:
                with tqdm(docs) as pbar:
                    pbar.set_description(pbar_desc)
                    return list(pool.map(fn, pbar))
        
    def clean_doc(self, doc):
        ''' Clean a document. '''
        import re
        clean_doc = re.sub('&amp;', '&', doc)
        clean_doc = re.sub(r'-+', ' ', clean_doc)
        clean_doc = re.sub(r'http[./:a-zA-Z0-9]+', ' ', clean_doc)
        clean_doc = re.sub(r' +', ' ', clean_doc) 
        return clean_doc

    def compile(self, json_fname = 'tweets.json'):
        ''' Load in the tweets from a json file, split into sentences 
            and count syllables.
    
            INPUT
                json_fname = 'tweets.json'
                    The json file containing all the tweets
        '''
        import json
        import os
        import spacy
        import pandas as pd
        from tqdm import tqdm
        from syllablecounter import load_model

        with open(os.path.join('data', json_fname), 'r') as file_in:
            tweets = json.load(file_in)
            tweets = [tweet['text'] for tweet in tweets]

        # Load SpaCy's English NLP model
        nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'ner'])

        # Tokenise tweets and pull out cleaned sentences
        tweets = self.map_docs(tweets, fn = nlp,
            pbar_desc = 'Splitting tweets into sentences')
        sents = [self.clean_doc(sent.text) for tweet in tweets 
            for sent in tweet.sents]

        # Count syllables
        counter = load_model()
        syls = self.map_docs(sents, fn = counter.predict,
            pbar_desc = 'Counting syllables', workers = 1)

        # Store the cleaned sentences and syllables
        self.sents = pd.DataFrame({
            'sentence': sents,
            'syllables': syls
            })

        # Remove blank sentences
        self.sents.dropna(inplace = True)
        self.sents = self.sents[self.sents['syllables'] > 0]

        # Save sentences
        self.sents.to_csv(os.path.join('data', 'sents.tsv', 
            index = False, sep = '\t'))

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

    tt = TrumpTweets()
    tt.compile()

    print('HAIKU 1:')
    print(tt.haiku())

    print('\nHAIKU 2:')
    print(tt.haiku())

    print('\nHAIKU 3:')
    print(tt.haiku())
