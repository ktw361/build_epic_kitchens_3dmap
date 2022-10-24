mkdir -p vocab_bins/

$(cd vocab_bins && wget https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin)
$(cd vocab_bins && wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin)