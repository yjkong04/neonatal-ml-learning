
*Goal: close the genuine theory gaps to "explain it" depth, not "derive it" depth. One focused week on the math alone, then build ML while keeping math going in parallel. This is calibrated to your actual target lanes (growth-stage applied-ML + FDE), NOT applied-scientist/PhD derivation rounds. If you find yourself prepping to derive a Hessian, you've over-scoped — stop.*

---

## The calibration (read this first, it governs everything below)

You marked everything blank except calculus. That's a real foundational gap — but the interview research narrows what "close it" means:

- **Linear algebra → conceptual only.** Explain what matrix multiplication *does*, what eigenvectors are, why PCA uses them. No derivations, no proofs. 3Blue1Brown is sufficient; you do NOT need a course on top of it.
- **Probability → the heaviest lift,** because it's where the applied-ML interview surface actually is (Bayes, MLE, bias–variance, why cross-entropy not MSE). Still conceptual generation, not measure theory.
- **Calculus → you have it.** Light refresh only on the ML-relevant bits (gradients, chain rule → backprop). Don't re-study calc.

**The test for "deep enough":** can you talk an interviewer through *why*, out loud, without notes? If yes, stop and move on. Depth past that is the credentialing reflex, not prep.

---

## Week 1 — Math-only sprint (~the focused week you asked for)

Realistic load: 2–3 hrs/day, or compress into fewer bigger blocks if your week is lumpy. The order matters: **probability gets the most time, linalg is lighter, calc is a touch-up.** Don't go linalg-first just because it feels like "the foundation" — your interview surface is probability-heavy.

### Days 1–3 — Probability & statistics (the heavy block)

Primary resource: **[StatQuest (Josh Starmer) on YouTube](https://www.youtube.com/c/joshstarmer)** — built for exactly this level, digestible, ML-aimed. Supplement with the relevant bits of **[Chip Huyen's *Introduction to Machine Learning Interviews*](https://huyenchip.com/ml-interviews-book/)** (free) for the interview framing. Practice questions live in the **[Chip Huyen question bank](https://huyenchip.com/ml-interviews-book/contents/8.1.2-questions.html)** and **[Khan Academy AP Statistics](https://www.khanacademy.org/math/ap-statistics/probability-ap)**.

Work through, in order, and for each one write a 2–3 sentence "explain it to an interviewer" note in your own words:

1. **Probability vs. likelihood** — what's the difference, and what does MLE actually maximize? (This trips up most people; nail it early.)
   - Watch: [Probability is not Likelihood — StatQuest](https://www.youtube.com/watch?v=pYxNSUDSFH4) + [Maximum Likelihood, clearly explained — StatQuest](https://www.youtube.com/watch?v=XepXtl9YKwc)
	   - area under the curve = P(x) = probability
		   - mean and stdev describe the shape of the curve
		   - modify area under a fixed distribution
		   - P (data | distribution)
	- likelihood is the reverse (given x in a specific curve, what is y)
		- modify distributions to find likelihood of y for fixed x
		- L (distribution | data)
		- maximum likelihood: optimal way to fit distribution to **data******
			- MLE: Maximum Likelihood Estimation
   - Practice: [Chip Huyen Q bank → "Probability" section](https://huyenchip.com/ml-interviews-book/contents/8.1.2-questions.html)

2. **Bayes' theorem** — state it, then do one concrete base-rate problem (disease test is the classic). Connect to why a rare condition + imperfect test gives surprising posteriors — directly relevant to your jaundice classifier's positive predictive value.
   - Watch: [Bayes' Theorem, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=9wCnvr7Xw4E)
   - Practice: [Khan Academy — Conditional probability exercises](https://www.khanacademy.org/math/ap-statistics/probability-ap/stats-conditional-probability/e/calculating-conditional-probability) — do 3–4 then swap the numbers for your jaundice classifier

3. **Distributions you'll actually cite** — Bernoulli, binomial, normal. Don't memorize a zoo; know these three and what they model.
   - Watch: [The Main Ideas behind Probability Distributions — StatQuest](https://www.youtube.com/watch?v=oI3hZJqXJuc) + [The Normal Distribution, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=rzFX5NWojp0)
   - Practice: [Khan Academy — Binomial distribution practice](https://www.khanacademy.org/math/ap-statistics/probability-ap) — search "binomial" in the unit

4. **Expectation & variance** — definitions, and variance as "spread." Enough to reason about, not derive.
   - Watch: [Covariance, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=qtaqvPAeEJY) (covers expectation + variance as lead-in)
   - Practice: [Khan Academy — Expected value practice](https://www.khanacademy.org/math/ap-statistics/probability-ap) — search "expected value"

5. **Bias–variance as a decomposition** — not the slogan. Where does each term come from, and what moves it (model complexity, more data). This is a top-5 most-asked concept.
   - Watch: [Bias-Variance Trade-Off, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=Unol0NZtuRo)
   - Practice: No calculation here — write your explain-it answer cold; then check it against the StatQuest video summary

6. **Covariance vs. correlation** — and why correlation is unitless. (Sets up PCA on Day 4.)
   - Watch: [Covariance, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=qtaqvPAeEJY)
   - Practice: [Chip Huyen Q bank](https://huyenchip.com/ml-interviews-book/contents/8.1.2-questions.html) — search for covariance/correlation questions

7. **p-values & confidence intervals** — what they precisely mean and, more importantly, what they do NOT mean.
   - Watch: [P Values, clearly explained — StatQuest](https://www.youtube.com/watch?v=5Z9OIYA8He8) + [Confidence Intervals, Clearly Explained — StatQuest](https://www.youtube.com/watch?v=TqOeMYtOc1w)
   - Practice: Write out 3 "this is NOT what a p-value means" statements. This is the actual interview question.

**End-of-block check:** can you explain bias–variance and probability-vs-likelihood cold? If not, those two specifically are worth a second pass before moving on.

### Days 4–5 — Linear algebra (conceptual, lighter)

Primary resource: **[3Blue1Brown — "Essence of Linear Algebra"](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)** (~3 hrs of video total, free, YouTube). This is the whole lin alg plan. Its entire purpose is visual conceptual intuition, which is exactly — and only — what you need. 

Watch in order, and again write a short "explain it" note for each:

1. **Vectors and linear combinations** — what a vector *is* in this context.
   - [Chapter 1: Vectors](https://www.youtube.com/watch?v=fNk_zzaMoSs)
2. **Linear transformations & matrix multiplication** — the key reframe: a matrix is a transformation of space, multiplication is composing them. This is the single most useful intuition on the list.
   - [Chapter 3: Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE)
3. **Determinant** — what it means for area/volume; what a zero determinant says (singular, non-invertible).
4. **Rank, column space, null space** — conceptually, what rank tells you.
5. **Eigenvectors & eigenvalues** — definition and the geometric picture (vectors that don't change direction under the transformation).
6. **Then connect to PCA yourself:** PCA finds the directions of maximum variance = eigenvectors of the covariance matrix, ordered by eigenvalue. Be able to say *that sentence* and what you're projecting onto. That's the interview bar.

**Do NOT** go learn SVD derivations, matrix calculus, or proofs. If a resource starts proving things, you've left your scope.

### Day 6 — Calculus touch-up (light) + optimization glue

You have calc, so this is a refresh of only the ML-relevant pieces:

1. **Gradient = direction of steepest ascent;** gradient descent moves opposite it to reduce loss. Be able to say why.
   - Watch: [Gradient Descent, Step-by-Step — StatQuest](https://www.youtube.com/watch?v=sDv4f4s2SB8)
2. **Chain rule → backprop** is just repeated chain rule through layers. You don't need to hand-derive backprop, just understand it's the chain rule mechanized.
   - Watch: [Backpropagation Details Pt. 2: Going bonkers with The Chain Rule — StatQuest](https://www.youtube.com/watch?v=GKZoOHXGcLo)
3. **Convex vs. non-convex** — you already have the instinct (convexity guarantees the global optimum). Pin the correction: convex = bowl = global *minimum*, which is what loss minimization wants. Concave = global max.

### Day 7 — Consolidate + self-quiz

- Run back through the original diagnostic question list out loud. Re-mark each: fluent / shaky / blank.
- Anything still shaky → one targeted re-watch or re-read, not a whole re-study.
- Write your one-page gap map: what's now fluent, what's still soft. This is the artifact your plan asked for.

**Realistic outcome after week 1:** you won't be "done" with math — you'll be at "can explain the core concepts, still shaky on a few." That is the correct stopping point to start building. Math past here happens *in parallel*, triggered by what you actually hit while building.

---

## Week 2 onward — Build ML, math in parallel

Now the math stops being a standalone sprint and becomes **just-in-time learning** attached to the MLOps "deploy Kramer" project from your job-search plan. This is the better way to retain it anyway — concepts stick when you hit them in real work, not in the abstract.

### The build spine (from your plan — Phase 1 MLOps spike)

This is the actual work; the math rides along:

1. **Serve** — wrap the Kramer model in a FastAPI prediction endpoint.
2. **Containerize** — Docker.
3. **Monitor** — basic logging/metrics on the served model.
4. **Version** — DVC, building on your existing W&B tracking.
5. *(stretch)* simple front-end that calls the API → demoable.

### How math attaches to the build (just-in-time triggers)

| When you're working on… | The math concept it pulls in |
|---|---|
| The model's output / prediction endpoint | Logistic regression as log-odds; what the probability output means |
| Monitoring metrics, evaluating the classifier | ROC/AUC, precision/recall, why **accuracy lies on imbalanced data** — your jaundice case directly |
| Any threshold / calibration discussion | Bayes / base rates / positive predictive value |
| If/when you touch model internals or retraining | Gradient descent, the loss function, regularization (L1/L2) |
| Explaining *why* the model does what it does | Bias–variance in the concrete |

The discipline: when the build surfaces a concept you're shaky on, **that's** when you go back to the StatQuest/3B1B segment for it — 20 minutes, targeted, then back to building. Don't pre-learn everything; let the work pull the theory.

### Cadence (so this survives Corvita + the degree)

Per your job-search plan's weekly rhythm:

- **One block/week** on the build (this *is* the repo/project block).
- **A small ongoing math habit** — a StatQuest video or one ISLR section per week, plus the just-in-time lookups. Not a separate grind.
- **ISLR** becomes readable now — pull the relevant chapter (logistic regression, regularization, resampling/cross-validation) when the build touches it, rather than reading cover-to-cover.

---

## Resources (all free)

- **StatQuest** — [Channel](https://www.youtube.com/c/joshstarmer) | probability & ML stats, your primary for the heavy block.
- **3Blue1Brown — "Essence of Linear Algebra"** — [Full playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (~3 hrs) — the entire linalg plan.
- **Chip Huyen — *Introduction to Machine Learning Interviews*** — [Free online book](https://huyenchip.com/ml-interviews-book/) | [Questions bank](https://huyenchip.com/ml-interviews-book/contents/part-ii.-questions.html)
- **ISLR (*Introduction to Statistical Learning*)** — free PDF; use as a just-in-time reference in Week 2+, not a Week-1 read-through.
- **3Blue1Brown — "Essence of Calculus"** — only if the Day 6 touch-up reveals more rust than expected.
- **Khan Academy — Statistics & Probability** — [AP Statistics practice](https://www.khanacademy.org/math/ap-statistics/probability-ap) — best source for calculation problems on Bayes, distributions, expectation.

---

## Guardrails (the failure modes for *this* specifically)

1. **Over-scoping the math into a stall.** The biggest risk. "I need to learn all of math first" is the credentialing reflex wearing a math hat. You need *enough to explain*, then you build. The deployed model + your regulatory angle is what gets you read — the theory just keeps you from falling apart on "why" follow-ups.
2. **Don't let the math week eat the warm-network work.** Your job-search plan flags that network activation quietly dies when a week compresses. This math sprint is *one* week and it's front-loaded precisely so it's done before Phase 2's relationship work needs your attention. Don't let "I'm still doing math" become the reason the network phase slips.
3. **Don't chase the derivation tier.** Proof-heavy rounds (Hessians, k-means centroid proofs) are applied-scientist/PhD territory, a different lane than your warm core. If your prep drifts there, you're studying for a job you're not applying to.
4. **Calc is done — don't re-study it.** You have it. Touch-up only.
