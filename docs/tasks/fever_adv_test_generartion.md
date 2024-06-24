# Reasoning
**What is NLI?** ➡ is the task of determining whether a hypothesis $H$ is true (entailment), false (contradiction) or undetermined (neutral) given a certain premise $P$ (text). For example

$P$ : A man is waiting at the table of a restaurant
- $H_1$ : A man waits to be served his food **Entailmet**
- $H_2$ : A man is looking to order a sandwich **Neutral**
- $H_3$ : A man is waiting in line for the bus **Contradiction**

**Important notes** ➡ Note that task is sensitive to sentence order such that model should receive `[CLS] P [SEP] H`. Moreover NLI is not a task for evaluating truthfulness of $H$ given $P$ but test the soudness of $H$ given $P$

**Standard Approach** ➡ Treating NLI as 3-class classification task. Tranformer based architectures such as DeBERTa, RoBERTa are *de facto* standards in NLI

**Advanced Approaches** ➡ On different datasets such as [FEVER](https://fever.ai/resources.html), Semantic Role Labeling have been successful check out [Task Leaderboard](https://paperswithcode.com/sota/fact-verification-on-fever)

For our case at least we need to do following ➡ We should format the samples before forwarding into tokenizer then to transformer as `[CLS] P [SEP] H`

---

**Our Task** ➡ Dealing with Adversarial-NLI, more complex and challenging version of reggular NLI task. An adversarial sample is a sample generated to be particularly complex and intended to fool models trained on the normal distribution of the original NLI task. Note that adversarial samples are still sound and compliant to the task descriptions.

Besides **custom version of FEVER**, we've provided an **extra test set (the adversarial test set)**. To perform well on extra test data **we need a more robust system**

In this homework, you will fine-tune a model to solve plain NLI task over a custom downsampled FEVER dataset and see how the performances changes when testing on the extra test set.

Then, you are asked to augment part of the training
data and then check whether this leads to improved
performances or not.

---

**Explicit Semantics for More Robust NLI** ➡ As part of this homework we are asked to generate new samples from a subset of the training data of the NLI dataset. Such new exmaples have to be **adversarial** so more complex than the original samples (**When altering the original samples, do check that the resulting samples are sound!**). Here are some recommended techniques:

- <font color='pink'>Word Sense Disambiguation (WSD)</font> can be useful for this, process of identifying sense of a word in a sentence. We will rely on [AMuSE-WSD Orlando et al. 2021](https://aclanthology.org/2021.emnlp-demo.34/
- <font color='orange'>Semantic Role Labeling (SRL)</font> is the process that assign labels to words or phrases in a sentence that indicates their semantic role, such as that of an agent, goal, or result. Annotation of sentences present in our NLI training subset. To do this we will rely on [InVeRo-SRL Conia et al. 2020](https://verbatlas.org/search?word=il+gatto+rincorre+il+topo)

<font color='pink'>Given the WSD annotations, we can change the pointed sense with its hypernym, hyponym or synonym (usually, no lablel change on sample), or by one of its antonyms

- *A word w is the hypernym of a target word x if the meaning of w includes the meaning of x, which is more specific (e.g., animal is a hypernym of elephant)*
- *A word w is the antonym of a target word x if the meaning of w is the opposite of x (e.g., hot and cold)*
- *A word w is a synonym of a target word x if it means exactly or nearly the same as x (e.g., small and little)*</font>

<font color='pink'>For this we can use WordNet, for more information it is good idea to look at WordNet notebook, **we are free (and encouraged!) to use also other relationships to augment the data**</font>

<font color='pink'>Examples for possible augmenting data with</font>
- <font color='pink'>Hypernym substitution:
  - $P$ : The cat is running $H$ : The cat is moving quickly $L$ : ENTAILMENT
  - $P$ : The cat is running $H'$ : The animal is moving quickly $L$ : ENTAILMENT</font>

  <font color='pink'>Animal is the hypernym of cat, by substituting cat with animal we obtain another valid pair which preservers the ENTAILMENT relationship. **IT IS VERY IMPORTANT THAT PAIR WE GENERATE MUST NOT VIOLATE PROVIDED NLI LABLE E.G ENTAILMENT ➡ NEUTRAL**</font>


- <font color='pink'>Synonym substitution:
  - $P$ : The kitten is running $H$ : The kitten is moving quickly $L$ : ENTAILMENT
  - $P$ : The kitten is running $H'$ : The kittie is moving quickly $L$ : ENTAILMENTM</font>

  <font color='pink'>Here kittie is synonym of kitten, again replacing kitten with kitte preserves the entailment relationship among the two sentence</font>

- <font color='pink'>Antonym subsitution:
  - $P$ : The tall man is dancing $H$ : The tall man is moving rhytmically $L$ : ENTAILMENT
  - $P$ : The tall man is dancing $H'$ : The short man is moving rhytmically $L$ : CONTRADICTION</font>

<font color='pink'>**ADDITIONAL NOVELTIES AND IDEAS WILL BE REVARDE ON OTHER SEMANTIC-BASED TRANSFORMATIONS THAT LEAD TO VALID NLI INSTANCES!!!!!!!**</font>

<font color='orange'>**Semantic Role Labeling (SRL)** is the second advised approach for augmenting the data. Agent-Patient swap, given the SRL graph we can invert semantic roles **(such as AGENT and PATIENT)**

If the sentence is complex you can rely on the **SRL graph structure**, modifying it to create a more complex example, maintaining or changing the class of the training sample.</font>

<font color='orange'>An example would be, consider sentence **"Alice (AGENT) destroyed (DESTROY) Bob (PATIENT) at chess (ATTRIBUTE)"** ➡ **"Bob (AGENT) destroyed (DESTROY) Alice (AGENT) at chess (ATTRIBUTE)"**</font>

Above is two possible approach for augmenting with <font color='pink'>WSD</font> and <font color='orange'>SRL</font>. **The adversarial test set is made using a mix of difference techniques** This means **using semantics alone may or may not prove beneficial we are encouraged to try other approaches and see what works best!!!!!!**

**A Suggestion** is starting by modifying the hypotheses as they are shorter than the permises and so simpler to modify. **We must try mix of changes** to make more complext training set.

We can use **external dataset** for augmenting and making a more challening training set however **WE CANNOT TRAIN OUR MODEL WITH THEM ONLY ADDITIONS/AUGMENTATIONS TO TRAINING SET FROM EXTERNAL DATASET**  

We may need to use extra tools like, **POST tagging library to implement our ideas**

Below are possible approaches to augment the data using inference of entites in the training data

- Numerical inferences: e.g., inferring dates and ages from numbers
- Reference inferences: e.g., coreferences between pronouns and forms of proper names
- Names inferences: e.g., leveraging the gender information in the proper names
- Syntax inferences: e.g., conjunctions, negations, cause-and-effect, comparatives, superlatives
- Tricky inferences: e.g., wordplay, linguistic strategies such as syntactic transformations/reorderings

---

**Architectural changes for a more robust NLI** ➡ While we can use the additional information provided by the semantic annotations for anything you may think of, we could also try to include it to the model by changing the architecture. **We are not supposed t rethink the Transformer architecture (obviously)** but we can for example:

- add some modules to leverage those information
- use additional embedding coming from WSD, SRL, POS, etc.
- use an ensemble of models
- search the literature for adversarial NLI to get inspiration
- …experiment!


