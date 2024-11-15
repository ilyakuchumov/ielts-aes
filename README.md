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

task
scorer
rewriter - next steps
final solution plan


torch and g5 instances
https://stackoverflow.com/questions/70022761/pytorch-version-on-g5-xlarge


https://aws.amazon.com/ru/ec2/instance-types/
p2large - 1$ but v100 12gig ram - not enough ram?
p2.8xlarge - 7$ - too expensive
too expensive? - p3.2xlarge - 3$  16gig ram

3$ = 7 days of 24/7 working
1$ = 21 days of 24/7 working

g5.xlarge - 1$ A10G 16 gig ram - seems good
1.2$ g5.2xlarge 32 gig ram - seems good
1.8$ g5.4xlarge - seems good



https://docs.google.com/document/d/1LPPy4-HMmWqt8bj0Sv-ZaFQMXpO9gO6W/edit
/Users/ilyakuchumov/.ssh/gpu_key1.pem
ssh -i "/Users/ilyakuchumov/.ssh/gpu_key1.pem" ubuntu@ec2-3-239-96-176.compute-1.amazonaws.com

reward = band(new_essay) - band(old_essay) - lambda * chars_diff

lambda ~ 0.01 (every 100 chars costs 1 additional band point)

final decision 

idea
1) better if humans will score (seems that costs ~1$ per essay)
2) we want to change essay if we can improve score significantly
3) we don't want to change essay and decrease score

formal metric will be something like
1) -10 * share(score(new_essay) + 0.5 <= score(old_essay)) 
    + share(score(new_essay) > score(old_essay) and diff(new_essay, old_essay) < 100)

------------------------------------------

==== training ==== 

1. test few shot as a baseline?

2. fine-tuning

base algorithm is RLHF (https://aws.amazon.com/what-is/reinforcement-learning-from-human-feedback/?nc1=h_ls)

1) pretrained model 
2) loop (train scorer -> improve policy)

LoRA (Low Rank Adaptation) and QLoRA (Quantized LoRA), where pre-trained models are loaded to GPU
as quantized 8-bit and 4-bit weights

This takes about 16 hours on a single GPU and uses less than 10GB GPU memory;
changing batch size to 8/16/32 will use over 11/16/25 GB GPU memory.

different trainers in TLR (Transformer Reinforcement Learning)

general in https://huggingface.co/docs/trl/quickstart and https://huggingface.co/docs/trl/how_to_train
1) create PPO trainer (in quickstart this is PPOTrainer)
2) generate batch and score for it
3) making one step ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
4) model.save_pretrained("my-fine-tuned-model-ppo") 

1) https://huggingface.co/docs/trl/sft_trainer
   needs existing answers in data set?
2) https://huggingface.co/docs/trl/ppo_trainer   <---- seems the best option to start
   needs reward function and optimizing agent step by step
3) https://huggingface.co/docs/trl/dpo_trainer 
   needs pairs of (positive, negative) examples and than train a model to generate better options

=== plan ===
1) take llama 11B, load it on AWS (probably it will be better to start with 3B because of speed)
2) take some prompt and generate answers, check them as baseline
3) try ppo_trainer with LORA adapter to fine-tune several iterations

=== alternatives ===
1) 3B model or QLORA adapter in case of lack of memory
2) try 90B model in case of bad quality






=== links ===
- https://www.llama.com/
- https://www.llama.com/docs/how-to-guides/fine-tuning/
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/trl/dpo_trainer
- https://discuss.huggingface.co/t/supervised-fine-tuning-trainer-custom-loss-function/52717

