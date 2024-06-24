## We need to reframe the give task from different campaign as instructed in the `README.md` 
## files foud in the related task folders. Given datasets from these two different datasources 
## SemEval and EVALITA is not suited for LLM testking we need to reframe them, making them suitable 
## for evaluating LLMs
## 
## Reframe each task so that an LLM can receive a sentence and a prompt as input and produce a 
## single answer as output. There are multiple different tasks that we can get, but the general 
## the structure of the dataset should follow the below format in;
## 
## ```json
##   // dataset.jsonl
##   {
##     "sentence": str,
##     "answer" : list[str],
##     "label" : int
##   }  
## ```
## 
## Prompt engineering part is where the actual value might come in, we need to create at most 
## 5 prompts for each given tasks and provide them in different file named `prompt.json`in the 
## following format.
## 
## The meticulous design of the input provided to an LLM to solve a task. The aim is to 
## improve the performance and manage the behavior of LLMs. Along with the dataset 
## reformatting you have to provide different prompts for each of defined task.
## 
## ```json
##   //prompt.json
##   {
##       "prompt": str
##   }
## ```
## 
## Datasets assigned ↪ 8, 15 \
## Distractor ↪ 23
## 
## Then using the reframed task, an LSTM model is trained with additional baselines. 
## Furthermore we then fine-tune a transformer-based model on custom dataset derived from 
## FEVER for NLI. The model's performance then evaluated on an adversarial test set, 
## specificailly designed to be challenging

class Reframer:
    def __init__(self, source):
        self.source = source


class SardiStanceReframer(Reframer):
    pass

