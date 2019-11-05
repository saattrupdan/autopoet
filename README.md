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
3. Process the GRU outputs through two time-distributed dense layers each with 512 hidden units followed by ReLU functions
4. Finally project the outputs from the dense layers down to a single dimension across time, outputting a sequence of numbers $\langle x_1,\dots,x_n\rangle\in[0,1]^n$ of the same length as we started with
5. To get the syllable count, we simply sum up the probabilities and round to the nearest integer

This model currently achieves a 95.34% validation accuracy.

The reason why we sum up the *probabilities* in point (5), rather than rounding the probabilities first, is to deal with the situation where the model is unsure whether two consecutive characters begin a new syllable. These will have probabilities ~50% and so will constitute a single syllable rather than two. 
