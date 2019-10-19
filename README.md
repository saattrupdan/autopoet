# AutoPoet <img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/autopoet_data/logo.png" width="30" height="30" alt="Logo of quill pen"/>

Build poems from text sources. Data files related to this repo can be found here:

<p align = 'center'>
  <a href = 'https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/autopoet_data/'>
    https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/autopoet_data/
  </a>
</p>

## Todos

- [x] Fetch Trump tweet data
- [x] Generate vocabulary from Trump tweets
- [x] Build baseline model to count syllables in English words
- [ ] Optimise syllable counter
- [ ] Use model to build Haikus from Trump tweets
- [ ] Enable working with live tweets
- [ ] Enable working with other text sources
- [ ] Build progressive web app that generates poems

## Syllable data

The syllable counter is trained on the Gutenberg syllable corpus, consisting of ~180,000 English words split into syllables. The `process_gutsyls` notebook converts these into a format which is more convenient for our purposes. The raw dataset can be freely downloaded here:

<p align = 'center'>
  <a href = 'http://www.gutenberg.org/files/3204/'>
    http://www.gutenberg.org/files/3204/
  </a>
</p>
