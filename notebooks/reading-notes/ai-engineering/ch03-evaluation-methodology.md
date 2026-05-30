# Ch. 3 — Evaluation Methodology

**Sections:** Challenges of evaluating foundation models · Language modeling metrics (entropy, perplexity) · Exact evaluation · AI as a judge · Comparative evaluation

---

## Key concepts
- evaluation checks if the model is accurate and secure
##### traditional ML models vs. foundation models
- traditional ML: close-ended, specific tasks
    - only gives expected outputs
- foundation models do open-ended tasks
    - multiple possible ouputs
    - model details (model architecture, training data, training process) generally hidden

- important to have solid *evaluation benchmarks*
    - ex. GLUE, Super-GLUE, NaturalInstructions, SuperNaturalInstructions, MMLU-Pro (most recent)
    - evaluation also discovers new tasks a model could potentially do
    - evaluation lags behind algorithm training, modeling and training, AI orchestration in the whole scheme of AI research
    - evaluation methods depend on the model's needs; not a one size fits all
- foundation models are built off language models
    - model learns from statistical information and predicts what comes next in the training data
        - better model, lower training cross entropy
        - if data is close to a model's training data -> model performs better

#### Entropy 
-> how much info on average a token carries
- more entropy = more info = more difficult to predict next word in a language model

#### Exact Evaluation
- exact vs. subjective evaluation
    - ie. MCQ versus essay 
    - AI is a subjective evaluator dependent on model and prompt
    - closed-ended (NOA) vs open-ended (LLMs)
- functional correctness tests a system based on intended functionality
    - does the model do what it's supposed to
    - defining "supposed to" is the hard part
    - *execution accuracy*
- code samples *k* code samples
    - problem is solved if all of *k* passes test cases
        - shown as *pass@k* score
- whenever there's a measurable outcome, functional correctness can be used
- alternate approach: evaluate against reference data
    - reference responses: *ground truths* or *canonical responses*

- *embedding* -> numerical representation of the original data
    - a vector. not limited to text
    - ex. transformer architecture generates embeddings to transform inputs to vectors
    - close similarity is measured by things like cosine similarity
    - embeddings are used in classification, topic modeling, RAG, etc.
    - *multimodal embedding space* represents different types of data (ex. image AND text)
        - need to use joint embeddings (ex. CLIP, ULIP, ImageBind)
        - goal: get the image embedding close to the corresponding text embedding
- AI can be a judge for evaluation models
    - fine-tuned models specifically for evaluation tasks (ex. Prometheus, Skywork-Critic)
    - humans calibrate the AI judge instead of replacing
        - we still need to catch hallucinations
    - the AI for evaluation can't be the same as AI for judge
        - this is called *model collapse* in training and *correlated blindspots* in evaluation
- AI as a judge works without reference data
    - criteria: correctness, repetitiveness, toxicity, wholesomeness, hallucinations, and more
- AI can evaluate a response + explain its decision
- how to prompt an AI judge
    1. give it a task to perform (ex. evaluate the relevance between a generated answer and question)
    2. criteria. should be detailed
    3. scoring system. (ex. classification, discrete/continuous numerical values)
- one reason to use a weaker model is for cost and latency
    - the stronger model can just edit the weak one's response
    - self-evaluation or self-critique for models to check if their own repsonse is correct
    - ex. writing a song versus judging a song

#### Ranking Models with Comparative Evaluation
- for evaluation, score doesn't matter. you're trying to find which model is right
    - ranked by pointwise evaluation or comparative evaluation
- pointwise evaluation: evaluate each model **independently** and rank by scores
- comparative evaluation: evaluate models **against each other** and make a ranking
    - good for subjective responses
- evaluation shouldn't always be based on preference, but sometimes on correctness
- comparative evaluation is NOT A/B testing
    - 

## NOA-AI connection
- NOA is a traditional ML system since there's a specific purpose and task
    - multiple specialist CNNs trained on one modality
    - rule-based scorers for clinical indices (such as APGAR, Sarnat, HeRO)
    - fusion engine that combines this into one clinical signal
    - narrative layer for the alarm
- use clinical AI for safety-critical outputs while being supervised by humans
    - AI judges for throughput evaluation (does the model format outputs correctly? does it catch obvious catches?)

## Questions / follow-up
