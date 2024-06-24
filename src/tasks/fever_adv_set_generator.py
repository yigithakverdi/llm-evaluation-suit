## Augementing the FEVER dataset with adversarial examples
## using the WordNet relations
## 
## Below is an example of sample pretty printed structure in subset of fewer 
## dataset which is the dataet we'll be augmenting
##
## PREMISE
## 'Premise: Roman Atwood . He is best known for his vlogs , where he posts '
##  'updates about his life on a daily basis . His vlogging channel , `` '
##  "RomanAtwoodVlogs '' , has a total of 3.3 billion views and 11.9 million "
##  "subscribers . He also has another YouTube channel called `` RomanAtwood '' , "
##  'where he posts pranks .'
## Label: ENTAILMENT
## 
## For token index 0 WSD attributes are as follows: {'bnSynsetId': 'bn:00109913a',
##  'index': 0,
##  'lemma': 'roman',
##  'nltkSynset': 'roman.a.01',
##  'pos': 'ADJ',
##  'text': 'Roman',
##  'wnSynsetOffset': '2921569a'}
## 
## 
## For token index 1 WSD attributes are as follows: {'bnSynsetId': 'O',
##  'index': 1,
##  'lemma': 'Atwood',
##  'nltkSynset': 'O',
##  'pos': 'PROPN',
##  'text': 'Atwood',
##  'wnSynsetOffset': 'O'}
## 
## HYPOTHESIS
## ('Premise: Johnny Galecki . He is known for playing David Healy in the ABC '
##  'sitcom Roseanne from 1992 -- 1997 and Dr. Leonard Hofstadter in the CBS '
##  'sitcom The Big Bang Theory since 2007 .')
## Label: NEUTRAL
## For token index 0 WSD attributes are as follows: {'bnSynsetId': 'O',
##  'index': 0,
##  'lemma': 'the',
##  'nltkSynset': 'O',
##  'pos': 'DET',
##  'text': 'The',
##  'wnSynsetOffset': 'O'}
## 
## 
## For token index 1 WSD attributes are as follows: {'bnSynsetId': 'O',
##  'index': 1,
##  'lemma': 'Boston',
##  'nltkSynset': 'O',
##  'pos': 'PROPN',
##  'text': 'Boston',
##  'wnSynsetOffset': 'O'}
## 
## And here is the more formal structure of the dataset
## {
##   "id": "150448",
##   "premise": "Roman Atwood . He is best known for his vlogs , where he posts updates about his life on a daily basis . His vlogging channel , `` RomanAtwoodVlogs '' , has a total of 3.3 billion views and 11.9 million subscribers . He also has another YouTube channel called `` RomanAtwood '' , where he posts pranks .",
##   "hypothesis": "Roman Atwood is a content creator.",
##   "label": "ENTAILMENT",
##   "wsd":
##     {
##       "premise":
##         [
##           {
##             "index": 0,
##             "text": "Roman",
##             "pos": "ADJ",
##             "lemma": "roman",
##             "bnSynsetId": "bn:00109913a",
##             "wnSynsetOffset": "2921569a",
##             "nltkSynset": "roman.a.01",
##           },
##           ...,
##         ],
##       "hypothesis":
##         [
##           {
##             "index": 0,
##             "text": "Roman",
##             "pos": "PROPN",
##             "lemma": "Roman",
##             "bnSynsetId": "O",
##             "wnSynsetOffset": "O",
##             "nltkSynset": "O",
##           },
##           ...,
##         ],
##     },
##   "srl":
##     {
##       "premise":
##         {
##           "tokens": [{ "index": 0, "rawText": "Roman" }, ...],
##           "annotations":
##             [
##               {
##                 "tokenIndex": 4,
##                 "verbatlas":
##                   {
##                     "frameName": "COPULA",
##                     "roles":
##                       [
##                         { "role": "Theme", "score": 1.0, "span": [3, 4] },
##                         { "role": "Attribute", "score": 1.0, "span": [5, 22] },
##                       ],
##                   },
##                 "englishPropbank":
##                   {
##                     "frameName": "be.01",
##                     "roles":
##                       [
##                         { "role": "ARG1", "score": 1.0, "span": [3, 4] },
##                         { "role": "ARG2", "score": 1.0, "span": [5, 22] },
##                       ],
##                   },
##               },
##               ...,
##             ],
##         },
##     },
## }
import threading

def import_hf():
    global load_dataset
    from datasets import load_dataset

    print("[DEBUG] Huggingface datasets imported")

def import_nltk():
    global nltk, wn, api
    import nltk
    from nltk.corpus import wordnet as wn
    import gensim.downloader as api
    print("[DEBUG] NLTK imported")

hf_thread = threading.Thread(target=import_hf)
nltk_thread = threading.Thread(target=import_nltk)

hf_thread.start()
nltk_thread.start()

hf_thread.join()
nltk_thread.join()

## Base imports
print("[DEBUG] Importing base libraries")
from itertools import chain
import pprint as pp
import os
import random

## Huggingface imports
# print("[DEBUG] Importing Huggingface datasets")
# from datasets import load_dataset

## NLTK imports
# print("[DEBUG] Importing NLTK")
# import nltk
# from nltk.corpus import wordnet as wn




## NLTK wordnet download
if(os.path.exists("/home/yigitwsl/nltk_data") == False):
    print("[DEBUG] Downloading NLTK wordnet")
    nltk.download('wordnet')
else:
    print("[DEBUG] NLTK wordnet already downloaded")

## Cache dir
cache_dir = "/mnt/c/Users/hakve/workspace/github/llm-evaluation-suit/data"

## Load datasets
dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli", 
                                cache_dir=cache_dir)

if(os.path.exists("/home/yigit/gensim-data") == False):
    print("[DEBUG] Downloading GloVe embeddings")
    wv = api.load('glove-wiki-gigaword-100')
else:
    print("[DEBUG] GloVe embeddings already downloaded")
    wv = api.load('glove-wiki-gigaword-100')

## Function for automatically mapping given lemmas and pos tags to the 
## WordNet relations
def get_wordnet_relations(lemma, pos):
    synsets = wn.synsets(lemma, pos=pos)
    if not synsets:
        return [], [], [], []

    # print(f"Synsets -> {synsets}")
    # synset = synsets[0]  # Using the first synset for simplicity
    # synonyms = set(lemma for syn in synsets for lemma in syn.lemma_names())
    # hypernyms = set(lemma for syn in synset.hypernyms() for lemma in syn.lemma_names())
    # hyponyms = set(lemma for syn in synset.hyponyms() for lemma in syn.lemma_names())
    # antonyms = set(lemma.antonyms()[0].name() for lemma in synset.lemmas() if lemma.antonyms())

    # return synonyms, hypernyms, hyponyms, antonyms  
    synonyms = set()
    hypernyms = set()
    hyponyms = set()
    antonyms = set()
    
    for synset in synsets:
        synonyms.update(synset.lemma_names())
        hypernyms.update([lemma_name for hyper in synset.hypernyms() for lemma_name in hyper.lemma_names()])
        hyponyms.update([lemma_name for hypo in synset.hyponyms() for lemma_name in hypo.lemma_names()])
        antonyms.update([antonym.name() for lemma in synset.lemmas() if lemma.antonyms() for antonym in lemma.antonyms()])
    
    return list(synonyms), list(hypernyms), list(hyponyms), list(antonyms)
    

## Using PoS tags and contextual relevance provided by methods of WordNet such as
## lowest_common_hypernyms, path_similarity, etc. to filter the relations
def filter_wordnet_relations(synonyms, hypernyms, hyponyms, antonyms, lemma, pos, original_text):
    pos_filter = []
    context_filter = []

    related_words = set(synonyms + hypernyms + hyponyms + antonyms)
    pos_filter = [word for word in related_words if wn.synsets(word, pos=pos)]    
    ## LC hypernyms filtering for the context, commented out for temporary, might be used later
    # for word in pos_filter:
    #     word_synset = wn.synsets(word, pos=pos)
    #     if(word_synset):
    #         ## Lowest common hypernym
    #         common_hypernyms = set()
    #         for synset1 in word_synset:
    #             for synset2 in wn.synsets(lemma, pos=pos):
    #                 common_hypernyms.update(synset1.lowest_common_hypernyms(synset2))
    #                 #
    #             # common_hypernyms.update(synset.lowest_common_hypernyms(wn.synsets(lemma, pos=pos)))
            
    #         if(common_hypernyms):
    #             print("\n"*3)
    #             print("Common hypernyms for {} and {} are:".format(word, lemma))
    #             print(common_hypernyms)
    #             context_filter.append(word)
    
    # relevance_replacements = [word for word in context_filter if word != lemma and word.lower() != original_text.lower()]
    # return relevance_replacements

    return pos_filter

def get_max_similarity(word, words):
    max_similarity = 0
    max_word = None
    for w in words:
        similarity = wv.similarity(word, w)
        if similarity > max_similarity:
            max_similarity = similarity
            max_word = w
    return max_word


## Function for augementing the sentence
def augement_sentence(sample, wsd_annotations):
    augmented_sentences = []
    content_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
    wn_pos_map = {'NOUN': wn.NOUN,
                  'VERB': wn.VERB,
                  'ADJ': wn.ADJ,
                  'ADV': wn.ADV
                  }
    
    print(f"Premise -> {sample['premise']}")
    print(f"Hypothesis -> {sample['hypothesis']}")
    
    premise_wds = wsd_annotations["premise"]
    hypothesis_wds = wsd_annotations["hypothesis"]

    for token_wds in hypothesis_wds:
        lemma = token_wds['lemma']
        pos = token_wds['pos']
        text = token_wds['text']
        
        ## If the PoS is in the content PoS, augmeinting word with WordNet relations 
        ## such as synonyms, hypernyms, hyponyms, antonyms
        if(pos in content_pos):
            print(f"Lemmma -> {lemma}, POS -> {pos}, Text -> {text}")            
            synonyms, hypernyms, hyponyms, antonyms = get_wordnet_relations(lemma, wn_pos_map[pos])
            # replacements = filter_wordnet_relations(synonyms, hypernyms, hyponyms, antonyms, lemma, wn_pos_map[pos], text)
            
            print(f"Token -> {text}")
            print(f"Synonyms -> {synonyms}")
            print(f"Hypernyms -> {hypernyms}")
            print(f"Hyponyms -> {hyponyms}")
            print(f"Antonyms -> {antonyms}")

            # replacements = synonyms | hypernyms | hyponyms | antonyms
            # replacements = [word for word in replacements if word != lemma and word.lower() != text.lower()]            
            
            if hypernyms:
                # replacement = random.choice(hypernyms)
                replacement = get_max_similarity(lemma, hypernyms)
                print(replacement)
                # augmented_sentence = sample['hypothesis'].replace(text, replacement)
                # augmented_sentences.append(augmented_sentence)

            print("\n"*2)
    
    pp.pprint(augmented_sentences)
    
sample = dataset["train"][1]
augement_sentence(sample, sample["wsd"])





