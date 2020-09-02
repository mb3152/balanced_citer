# Unbiased Citer
This repo calculates the race and gender (probabilistically) of the first and last authors for papers in your citation list (in the form of a .bib file) and compares your list to expected distributions based on a model that accounts for paper characteristics (e.g., author location, author seniority, journal, et cetera.) unrelated to race and gender. 

Requirements:
(1) ethnicolr and pybtex; when you run the code, it will install then on the fly for you if you do not have them.
(2) You also need a key from gender-api.com. This is free.
(3) a .bib file that contains the citations you want to analyze.

USAGE:
python unbiasedciter.py -authors 'Maxwell Bertolero Danielle Bassett' -bibfile '/path/to/my/main.bib' -gender_key 'my_gender_key'

This will create:
![Image](https://raw.githubusercontent.com//mb3152/unbiasedciter/master/race_gender_citations.png?raw=true)
