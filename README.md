# unbiasedciter
This repo calculates the race of authors in your citation list (in the form of a .bib file) and compares to expected distributions based on neuroscience papers published after 2016

The only non-stock (not in Anacondas) package you need to run this is ethnicolr. If you run the file, it will install this on the fly.

You also need a key from gender-api.com

USAGE:
python unbiasedciter.py -authors 'Maxwell Bertolero Danielle Bassett' -bibfile '/path/to/my/main.bib' -gender_key 'my_gender_key'

This will create:
![Image](https://raw.githubusercontent.com//mb3152/unbiasedciter/master/race_gender_citations.png?raw=true)
