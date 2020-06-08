# unbiasedciter
This repo calculates the race of authors in your citation list (in the form of a .bib file) and compares to expected distributions based on neuroscience papers published after 2016

the only non-stock (no in Anacondas) package you need to run this is ethnicolr
you can either "conda install -c soodoku ethnicolr" or "pip install ethnicolr"

USAGE:
python unbiasedciter.py -authors 'Maxwell Bertolero Danielle Bassett' -bibfile '/path/to/my/main.bib'
