# SardiStance - stance detection in Italian tweets (Dataset 8)

Homepage: http://www.di.unito.it/~tutreeb/sardistance-evalita2020/index.html

Download Link: https://live.european-language-grid.eu/catalogue/corpus/5245/download/

You have to subscribe a google form with your email you will recieve a password that you have to use to unzip on their

With this task proposal we would like to invite participants to explore features based on the textual content of the tweet, such as structural, stylistic, and affective features, but also features based on contextual information that documents not emerge directly from the text, such as for instance knowledge about the domain of the political debate or information about the user's community.

## Data

The data come in three `.csv` files, the `TRAIN_anonymized.csv` in the *development* folder contains the tweets and labels, while the `TEST_anonymized.csv` and `TEST-GOLD.csv` have to be merged to have tweets and labels. 

## Expected output

You have to solve only task A.

Create ```SardiStance-train.jsonl``` and ```SardiStance-test.jsonl```.

Each line in your output file must be a JSON object like the one below:

```JSON
{
    "text": str,
    "choices": list[str]
    "label": int
}
```

### Prompts

Create ```prompt.jsonl```.

In this file you have to report the prompts you designed for the task. 
Each line in your output file (1 line per prompt) must be a JSON object like the one below (max 5 lines in this file):

```JSON
{
    "prompt": str
}
```

## Deliver format

You have to format your data using JSON Lines standard.

## License

Creative Commons Attribution Non Commercial Share Alike 4.0 International


# PRELEARN (Prerequisite Relation Learning) (Dataset 15)

Homepage: https://sites.google.com/view/prelearn20
Datasets: https://live.european-language-grid.eu/catalogue/corpus/8084/download/

PRELEARN (Prerequisite Relation Learning) is a shared task on concept prerequisite learning which consists of classifying prerequisite relations between pairs of concepts distinguishing between prerequisite pairs and non-prerequisite pairs. For the purposes of this task, prerequisite relation learning is proposed as a problem of binary classification between two distinct concepts (i.e. a concept pair).

## Data

- *PRELEARN_DATASET*: A dataset built upon the AL-CPL dataset (Liang et al. 2018), a collection of binary-labelled concept pairs extracted from textbooks on four domains: data mining, geometry, physics and precalculus. 
- *ITA_prereq-pages*: a “Wikipedia pages file” containing the raw text of the Wikipedia pages referring to the concepts extracted using WikiExtractor on a Wikipedia dump of Jan. 2020.

You will find *-pairs* files where there are different pairs of concepts and a binary label that indicate the causality.

The data comes splitted in a train and test sets, for each task you have to mantain those splits.
The *prereq* contains wikipedia passages relative to the concepts in the *-pairs* files.

## Expected output

We ask you to create four different datasets, one for each domain, *data_mining*, *geometry*, *physics*, and *precalculus*.

### Subtask A - PRELEARN-data_mining

Create ```PRELEARN-data_mining-train.jsonl``` and ```PRELEARN-data_mining-test.jsonl```

Each line in your output file must be a JSON object like the one below:

```JSON
{
    "wikipedia_passage_concept_A": str,
    "concept_A": str,
    "wikipedia_passage_concept_B": str,
    "concept_B": str,
    "choices": list[str]
    "label": int,
}
```

### Subtask A - PRELEARN-geometry

Create ```PRELEARN-geometry-train.jsonl``` and ```PRELEARN-geometry-test.jsonl```

Each line in your output file must be a JSON object like the one below:

```JSON
{
    "wikipedia_passage_concept_A": str,
    "concept_A": str,
    "wikipedia_passage_concept_B": str,
    "concept_B": str,
    "choices": list[int]
    "label": int,
}
```

### Subtask A - PRELEARN-physics

Create ```PRELEARN-physics-train.jsonl``` and ```PRELEARN-physics-test.jsonl```

Each line in your output file must be a JSON object like the one below:

```JSON
{
    "wikipedia_passage_concept_A": str,
    "concept_A": str,
    "wikipedia_passage_concept_B": str,
    "concept_B": str,
    "choices": list[int]
    "label": int,
}
```

### Subtask A - PRELEARN-precalculus

Create ```PRELEARN-precalculus-train.jsonl``` and ```PRELEARN-precalculus-test.jsonl```

Each line in your output file must be a JSON object like the one below:

```JSON
{
    "wikipedia_passage_concept_A": str,
    "concept_A": str,
    "wikipedia_passage_concept_B": str,
    "concept_B": str,
    "choices": list[int]
    "label": int,
}
```

### Prompts

Create ```prompt-data_mining.jsonl```, ```prompt-geometry.jsonl```, ```prompt-physics.jsonl``` and ```prompt-precalculus.jsonl```.

In this file you have to report the prompts you designed for the task. 
Each line in your output file (1 line per prompt) must be a JSON object like the one below (max 5 lines in this file):

```JSON
{
    "prompt": str
}
```

## Deliver format

You have to format your data using JSON Lines standard.

# PoSTWITA (23 Distractor)

PoS tagging for Italian Social Media texts.

[Homepage](https://corpora.ficlit.unibo.it/PoSTWITA/)

## Task description

This is a standard Part-of-Speech tagging task, focusing on social media texts that are of a different nature compared to standard texts.

## Data

You should download the zip folder located in this European Language Grid [page](https://live.european-language-grid.eu/catalogue/corpus/7481/download/).

The files we are interested in are `goldDEVset-2016_09_05_anon_rev.txt` and `goldTESTset-2016_09_05_anon_rev.txt` in the `postwita/` folder

## Expected output

We expect one dataset for the PoS task.

Create `postwita-train.jsonl` (corresponding to the DEV original file) and `postwita-test.jsonl`.

Each line in your output file must be a JSON object like the one below:
```json
{
    "sentence_id": ...,
    "sentence": ...,
    "target_word": ...,
    "word_idx": ...,
    "choices": [...],
    "label": ...
}
```

### Distractors

We expect from you to design a strategy to include distractors among the choices, so you select three different uncorrect labels beside the correct one. These three labels must be challenging for the word in the given context.

### Prompts
Create `prompts.jsonl`, where each line of your file is a JSON object formatted as below:
```json
{
    "prompt": "..."
}
```

## Licence

This dataset is licensed according to [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
