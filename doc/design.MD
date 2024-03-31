## What is Homework is About?
We need to reframe the dataset given the related tasks such as if the task is about, "Word or sentence level tasks" we need to reframe the dataset accordingly

So the given datasets are not suited for testing generative LLMs our task is to **reframe each task** so that an LLM can receive a sentence and a prompt as input and produce a single answer as output

    Prompt --> LLM --> Answer

Structure of our dataset?? As above, a prompt and a corresponding answer.

By refreaming --> we have to reframe it to a multi-choice question answeirng (QA)

There are several tasks do we need to do every one of them??

%% Could be good practice to work on other datasets as well later on if not every task %%

**Prompt Engineering** ==> It is the other part of the homework, along with the dataset formatting we need to provide different prompts for each of defined tasks. An example is binary sentence level task, BoolQ is an example dataset for this

### Binary Sentence Level Task
**Step 1**: Transform the input item into a JSON containing passage, question the possible answers and the correct one:

```json
// FROM THIS
{
    "passage": "The passage text",
    "question": "The question text",
    "answer": "true"
}

// TO THIS
{
    "passage": "The passage text",
    "question": "The question text",
    "choices": ["true", "false"],
    "label": 0
}
```
**Step 2**: Come up with an effective prompt. For BoolQ it could be the following:

    Prompt: Given the paage {{passage}} answer the following question: {{question}}. The possible answers are: true, false.

There can be multiple prompts for the same task. Our job is to provide between 3 and 5 prompts for each task. **We need to motivate our deign choice**

### Categorical Sentence Level Task
Another example is categorical sentence level task CB, it consist in assigning a category to a given input sentence. ANLI is an example database for this

**Step 1**: Transform the input item into a JSON containing passage, question the possible answers and the correct one:

```json
// FROM THIS
{
    "premise": "The premise text",
    "hypothesis": "The hypothesis text",
    "label": "neutral"
}

// TO THIS
{
    "premise": "The premise text",
    "hypothesis": "The hypothesis text",
    "choices": ["neutral", "entailment", "contradiction"],
    "label": 1
}
```
**Step 2**: Come up with an effective prompt. For ANLI it could be the following:

    Prompt: Given the premise {{premise}} and the hypothesis {{hypothesis}} answer the following question: What is the relationship between the premise and the hypothesis? The possible answers are: neutral, entailment, contradiction.   

Note that the label is the index of the correct answer in the choices list. 

### Word Level Classification Task
In a word-level classification task we need to classify each word of an input sentence. Those particaluar cases needs to be reframed as multi-choice classification tasks

For example in the, POS tagging task, each word of input sentence is labeled with a specific tag such as NN (noun), VB (verb), JJ (adjective), etc. We need to reframe the task by creating a new sample for each word

```json
// FROM THIS
{
    "words" : ["The", "cat", "is", "black"],
    "tags" : ["DT", "NN", "VB", "JJ"]
}


//TO THIS
{
    "sentence_id": 0,
    "input": "The August deficit and the #2.2 billion surplus in July.",
    "target_word": "deficit",
    "word_idx": 2,
    "choices": ["DT", "NN", "VB", "JJ"],
    "label": 1
}
```

Then in this case the possilbe prompt would be as follows:

    Prompt: Given the sentence {{input}} and the word {{target_word}} at position {{word_idx}} answer the following question: What is the POS tag of the word {{target_word}}? The possible answers are: DT, NN, VB, JJ.

or

    Prompt: In the sentence {{input_sentence}}, which is the part of speech tag of the word {{target_word}}?

Note that, when we creat different samples from the same sentence, as in word-level classificaiton, we need to add a `sentence_id` field for each sample created form the same sentence. For eaxmple in the above example we will have same `sentence_id` for each sample derived from the same starting sentence

### Another Example - GSM8K for Generative Task
GSM8K | (Grade School Math 8K) is a dataset of high quality linguistically diverse grade school math word problems

```json
// FROM THIS
{
    "question": "Natalia sold clips to her friends. She sold 15 clips to each of her 2 friends. How many clips did she sell in total?",
    "answer": "Natalia sold 30 clips in total."
}

//TO THIS
{
    "input": "Natalia sold clips to her friends. She sold 15 clips to each of her 2 friends. How many clips did she sell in total?",
    "choices": ["Natalia sold 30 clips in total.", "Natalia sold 45 clips in total.", "Natalia sold 60 clips in total.", "Natalia sold 75 clips in total."],
    "label": 0
}
```

Then the prompt could be:

    Prompt: Given the following math word problem, answer the following question: How many clips did Natalia sell in total? The possible answers are: Natalia sold 30 clips in total., Natalia sold 45 clips in total., Natalia sold 60 clips in total., Natalia sold 75 clips in total.

