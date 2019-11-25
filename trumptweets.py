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
                sep = '\t', dtype = {'sentence': str, 'syllables': int})
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

    def remove_toks(self, doc, idxs: list) -> str:
        toks = [tok.text_with_ws for idx, tok in enumerate(doc) 
                                 if idx not in idxs]
        return ''.join(toks)
        
    def clean_doc(self, doc, remove_mentions = True, remove_urls = True):
        ''' Clean a SpaCy document. '''
        from nltk.tokenize.casual import _replace_html_entities
        import re

        mentions = [idx for idx, tok in enumerate(doc) 
                    if tok.text[0] == '@' and remove_mentions]
        urls = [idx for idx, tok in enumerate(doc) 
                if tok.like_url and remove_urls]
        doc = self.remove_toks(doc, mentions + urls)

        doc = _replace_html_entities(doc)
        doc = re.sub(r'[^a-zA-Z.,:;-_!"#%&/()=@£${\[\]}+~*^<> ]', '', doc)
        doc = re.sub(r' +', ' ', doc) 
        return doc.strip()

    def compile(self, json_fname = 'tweets.json', remove_mentions = True,
        remove_urls = True):
        ''' Load in the tweets from a json file, split into sentences 
            and count syllables.
    
            INPUT
                json_fname = 'tweets.json'
                    The json file containing all the tweets
                remove_mentions = True
                    Remove all mentions (e.g. @username) from the tweets
                remove_urls = True
                    Remove all urls from the tweets
        '''
        import json
        import os
        import en_core_web_sm as en
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        from functools import partial
        from syllablecounter import load_model

        with open(os.path.join('data', json_fname), 'r') as file_in:
            raw_tweets = json.load(file_in)
            tweets = [tweet['text'] for tweet in raw_tweets]
            ids = [tweet['id_str'] for tweet in raw_tweets]

        # Load SpaCy's English NLP model
        nlp = en.load(disable = ['tagger', 'ner'])

        # Tokenise tweets
        tweets = self.map_docs(tweets, fn = nlp,
            pbar_desc = 'Splitting tweets into sentences')

        # Clean tweets
        clean = partial(self.clean_doc, remove_mentions = remove_mentions,
            remove_urls = remove_urls)
        ids_sents = [(id, clean(sent)) for id, tweet in zip(ids, tweets)
                                       for sent in tweet.sents]
        ids = [id for id, _ in ids_sents]
        sents = [sent for _, sent in ids_sents]

        # Count syllables
        counter = load_model()
        syls = self.map_docs(sents, fn = counter.predict,
            pbar_desc = 'Counting syllables', workers = 1)

        # Store the cleaned sentences and syllables
        self.sents = pd.DataFrame({
            'id': ids,
            'sentence': sents,
            'syllables': syls
            })

        # Remove blank sentences
        self.sents['syllables'] = pd.to_numeric(
            self.sents['syllables'], 
            errors = 'coerce' # This converts non-numerics to NaN values
            )
        self.sents.replace('', np.nan, inplace = True)
        self.sents = self.sents[self.sents['syllables'] > 0].dropna()

        # Save sentences
        self.sents.to_csv(os.path.join('data', 'sents.tsv'), 
            index = False, sep = '\t')

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
                'Compile? (y/n)\n>>> '
            if input(query) == 'y':
                self.compile()

        if syllables:
            phrases = self.sents[self.sents['syllables'] == syllables]
            phrases = phrases['sentence']
        else:
            phrases = self.sents['sentence']

        try:
            phrase = random.choice(list(phrases))

            # Remove punctuation at the beginning or end of phrase
            phrase = re.sub('[\n―]', ' ', phrase)
            phrase = phrase.strip(' .,-… \'\"\”\“:&’_|')

            # Make the first letter in the phrase uppercase
            phrase = re.sub(r'^[a-z]', lambda m: m.group(0).upper(), phrase)

        except IndexError:
            phrase = f'<No phrase found with {syllables} syllables>'

        return phrase.strip()

    def haiku(self):
        ''' Get a random Trump haiku. '''
        line1 = self.rnd_phrase(5)
        line2 = self.rnd_phrase(7)
        line3 = self.rnd_phrase(5)
        return line1 + '\n' + line2 + '\n' + line3

if __name__ == '__main__':
    tt = TrumpTweets()
    tt.compile()
