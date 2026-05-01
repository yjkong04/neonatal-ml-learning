# Reading List

Curated papers and references that anchor each phase. Read alongside the coding work, not before it. The point is to ground the technique in a real problem.

---

## Foundational (Weeks 1–4)

- **Reyna et al. 2020** — "Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019." *Critical Care Medicine.* Open access. The benchmark for the anchor project.
- **Moorman et al. 2011** — "Mortality reduction by heart rate characteristic monitoring in very low birth weight neonates: A randomized trial." *The Journal of Pediatrics.* The original HeRO score paper.
- **Kramer 1969** — "Advancement of dermal icterus in the jaundiced newborn." *American Journal of Diseases of Children.* The original Kramer staging reference.

## Image CNNs (Phase 2)

- **He et al. 2015** — "Deep Residual Learning for Image Recognition." The ResNet paper. The architecture you'll fine-tune.
- **Tan & Le 2019** — "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
- **Esteva et al. 2017** — "Dermatologist-level classification of skin cancer with deep neural networks." *Nature.* The medical imaging benchmark that proved CNN transfer learning works in clinical contexts.

## Signal CNNs (Phase 3)

- **Hannun et al. 2019** — "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." *Nature Medicine.* The Stanford ECG paper.
- **Kiranyaz et al. 2021** — "1D Convolutional Neural Networks and Applications: A Survey." Good technical overview of 1D CNN architectures.
- **Faust et al. 2018** — "Deep learning for healthcare applications based on physiological signals: A review."

## Audio ML (Phase 4)

- **Hershey et al. 2017** — "CNN architectures for large-scale audio classification." (Google's AudioSet/VGGish paper.)

## Fusion & Calibration (Phase 5)

- **Guo et al. 2017** — "On Calibration of Modern Neural Networks." The paper that showed deep nets are systematically miscalibrated.
- **Lundberg & Lee 2017** — "A Unified Approach to Interpreting Model Predictions." The SHAP paper.
- **Rajkomar et al. 2018** — "Scalable and accurate deep learning with electronic health records." *npj Digital Medicine.* End-to-end clinical ML at scale.
- **Chen et al. 2021** — "Algorithmic fairness in healthcare." Survey on subgroup analysis methodology.
- **Sweet et al. 2023** — "European consensus guidelines on the management of respiratory distress syndrome." *Neonatology.* Useful clinical context for any respiratory work.

---

## Books (reference, not cover-to-cover)

- McKinney, *Python for Data Analysis* (3rd ed.) — pandas/NumPy reference, especially Chapter 11 on time-series.
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* — free online. Chapters on stationarity, decomposition, ARIMA.
- Goodfellow, Bengio & Courville, *Deep Learning* — reference only; do not read cover-to-cover.

## Courses

- fast.ai, *Practical Deep Learning for Coders* — Lessons 1–4 are the main path; Lessons 5–8 are good extension.
- Pranav Rajpurkar, *AI for Medical Diagnosis* (Coursera) — Phase 5 companion.
- Andrew Ng, *Machine Learning Specialization* (Coursera) — first three courses, audit free, only as a refresher if Phase 1 feels too sparse on theory.
