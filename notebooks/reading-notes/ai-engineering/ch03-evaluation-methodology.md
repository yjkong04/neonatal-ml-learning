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

## NOA-AI connection
- NOA is a traditional ML system since there's a specific purpose and task
    - multiple specialist CNNs trained on one modality
    - rule-based scorers for clinical indices (such as APGAR, Sarnat, HeRO)
    - fusion engine that combines this into one clinical signal
    - narrative layer for the alarm

## Questions / follow-up
