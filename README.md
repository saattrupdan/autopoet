# AutoPoet <img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/autopoet_data/logo.png" width="30" height="30" alt="Logo of quill pen"/>

Build poems from text sources. 

## Todos

- [x] Fetch Trump tweet data
- [x] Generate vocabulary from Trump tweets
- [x] Build baseline model to count syllables in English words
- [ ] Optimise syllable counter
- [x] Use model to build Haikus from Trump tweets
- [ ] Build progressive web app that generates poems
- [ ] Enable working with live tweets
- [ ] Enable working with other text sources

## Syllable model

A large part of this project was to develop a model that counts syllables in English words. 

The syllable counter is trained on a (slightly modified version of) the [Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg) syllable corpus, consisting of ~180,000 English words split into syllables. The `process_gutsyls` notebook converts these into a format which is more convenient for our purposes. The raw dataset can be freely downloaded [here](http://onlinebooks.library.upenn.edu/webbin/gutbook/lookup?num=3204), and the preprocessed versions used for this project can be found [here](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/autopoet_data/).

The model is a recurrent neural network that works at the character level, with the following rough architecture:

1. Embed the characters into 64-dimensional vectors
2. Process the characters through three bidirectional GRU layers, each having 2x128 = 256 hidden units
3. Process the GRU outputs through two timedistributed dense layers followed by a sigmoid function, outputting a binary sequence of the same length as we started with, with the 1's indicating whether the character starts a new syllable

This model currently achieves a 95.34% validation accuracy.
