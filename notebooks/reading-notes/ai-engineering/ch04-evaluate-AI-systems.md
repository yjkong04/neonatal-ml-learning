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
### Instruction-Following Capability

### Cost and Latency

---

## Model Selection

### Model Selection Workflow

### Model Build Versus Buy

### Navigate Public Benchmarks

---

## Design Your Evaluation Pipeline

### Step 1. Evaluate All Components in a System

### Step 2. Create an Evaluation Guideline

### Step 3. Define Evaluation Methods and Data

---

## NOA-AI connection

## Questions / follow-up
