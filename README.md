# ielts-aes

### Data

#### raw_hg
There are 2 datasets in the raw_hg folder. 
1. **data/chillies_gpt_eval_train.csv** and **data/chillies_gpt_eval_test.csv** - large data set with ~10k examples contains
   real task, essay from human, score from human and gpt-generated evaluation of this human score.

   (original link - https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation/tree/main)

2. **duwuonline_just_tasks.csv** - just ~2k questions and essays from humans without scores.
   
   (original link - https://huggingface.co/datasets/duwuonline/task2_ielts)

#### scorer_data
In this folder stores data for learning scorer iteration by
iteration in following format:

```
    id: int
    question: str
    essay: str
    score: float
    timestamp: str
    iteration: str
    source: str
```
