# Simple POS Tagger

A pos tagger that uses a simple but surprisingly effective algorithm. I have not seen this approach used before in literature surprisingly

The algorithm assumes each word's POS depends on the previous and latter word's POS. We initialize the pos tags with the most likely tag for that word. Then we sample from the conditional distribution for each word's tag given the previous and following tag, essentially performing gibbs sampling. The most common tag sequence is then returned

This model converges quickly and works well in practice because many words have a single tag that is correct roughly 98% of the time, thus these words can be treated as fixed without impacting accuracy and improving running speed
