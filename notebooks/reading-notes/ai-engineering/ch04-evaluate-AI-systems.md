# Ch. 4 — Evaluate AI Systems

**Sections:** Evaluation criteria · Model selection · Design your evaluation pipeline

---

## Evaluation Criteria
- defining the evaluation criteria before deployment is important for model to be useful
    - *evaluation-driven development*
- easy to use evaluation criteria when the outcome is measurable
- building more evaluation pipelines will help resolve AI adoption bottleneck
### Domain-Specific Capability
- domain-specific capabilities are limited by configuration and training data
- use domain-specific benchmarks to test the model's capabilities
    - exact evaluation
- code, for example, would be measured by functional correctness
- efficiency would be measured by runtime or memory usage
    - ex. BIRD-SQL
- for classification tasks, MCQ choices are the same for all questions
    - easy to create, verify, and evaluate
    - however, MCQ formatting (prompt) changes models' sensitivity 
    - good for knowledge (ex. is this correct, evalute this spreadsheet), bad for generating (ex. essays)
### Generation Capability
- NLG (natural language generation): evaluate the quality of open-ended text generation
- NLG includes several metrics like:
    - fluency: text grammatically correct? sounds natural?
    - coherence: logical structure of the whole text 
    required metrics can also change based on the task (ex. friendliness, positivity, factual consistency, etc.)
- hallucinations are actually desirable for creative tasks, not factual tasks
- factual consistency (obviously) is one of the most important metrics
    - local factual consistency
        - output evaluated against a given context
        - if the context is wrong, but output is right = factually inconsistent
        - important in limited scopes such as summarization, customer support chatbots, business analysis
    - global factual consistency
        - output evaluated against open knowledge 
        - important in broad scopes such as fact-checking, market research
- factual consistency is easier to verify against explicit facts
- important to gather what the facts are from reliable sources
    - depending on the source, a statement is factual or not
        - even a bogus statement can be factually consistent if the source says so
        -> *textual entailment*
        - entailment: hypothesis can be inferred from the premise (factual consistency)
        - contradiction: hypothesis contradicts the premise (factual inconsistency)
        - neutral: premise doesn't entail nor contradict the hypothesis (irrelevant)
- AI as a judge can evaluate factual consistency too
- specific scorers can be specialized
    - makes factual consistency classification-based
    - predicts entailment
### Instruction-Following Capability
- obviously, a core requirement
- a model can have domain-specific capability but bad instruction following
    - important to troubleshoot where the issue is coming from
        - instruction following vs. domain specific capability vs. generation capability
- outputs can be verified using follow up criteria
    - follow up criteria can be evaluated by human or AI evaluator
- *roleplaying* is a type of real-world instruction to help evaluate the model
    - gives users a character to interact with
    - as prompt engineering technique to improve quality of model's outputs
    - evaluate that the model stays in character

### Cost and Latency
- important consideration for a useful model
    - *Pareto optimization*: optimizing for multiple objectives
        - need to define non-negotiables and prioritize requirements
- more tokens to generate = higher latency
---

## Model Selection
- best model for your specific application is more important
    - starts with describing your criteria
1. figure out the best achievable performance
2. map models against cost and performance to account for cost and latency
### Model Selection Workflow
- hard attributes: what is impossible or impractical to change
    - model providers and your own policies decide this
- soft attributes: what you can and what you're willing to change
    - ex. accuracy, toxicity, factual consistency
    - balance optimism vs. being realistic
- evaluation workflow
    1. filter out models with irrelevant hard attributes (ex. commercial APIs vs. own models)
    2. use public data to narrow down promising models (ex. model quality, latency, and cost)
    3. run experiments with internal evaluation pipeline (our own criteria) to find the best model
    4. continuously monitor production to detect failure and collect feedback
- open weight: just publicly available
- open model: actually open data
    - flexible model usage (can retrain the model from scratch)
- open source models also need relevant licensing, each with their own conditions (importantly, commercial use)
#### Open Source Models vs. Model APIs
- inference service: service that hosts the model and receives user queries. also runs the models and returns it
    - model API: interface for users to interact w/. API of the inference service, for example
- so a model can be: open source, accessible via API, or both
    - APIs are guarded by a paywall, typically
    - accessible through model providers (OpenAI/Anthropic), cloud service providers (Azure, Google Cloud), or third-party (Databricks)
    - performance can vary slightly based on the API
- using an API provider risks leaking data to that provider to train their models 
---

## Design Your Evaluation Pipeline
- knowing good vs. bad outcomes

### Step 1. Evaluate All Components in a System
- different levels of evaluation: per task > per turn > and per immediate output
- end-to-end output is different from intermediate output from each component
    - problems at the end can be because of any intermediate step
    - evaluation method for each intermediate step is different
- ideal: evaluation per turn AND per task 
    - turns can be multiple steps (messages)
    - *turn-based evaluation*: evaluating the quality of each output
    - *task-based evaluation*: is the task completed
        - how many turns did it take
- 

### Step 2. Create an Evaluation Guideline

### Step 3. Define Evaluation Methods and Data

---

## NOA-AI connection
- we would do a blend of hosting the model ourself and using a model API (i think)
    - our infant proprietary model + existing APIs for LLMs and voice dictation
    - need to look at service-level agreements (SLA) when using another API
    - also, we need to run the model locally (need self-hosting model)
## Questions / follow-up
