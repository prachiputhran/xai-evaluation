# 🔍 Explainable AI Evaluation: SHAP + LIME

> **A comparative evaluation of model interpretability, explanation stability, and consistency across machine learning models and datasets.**

This project investigates the reliability of **Explainable AI (XAI)** techniques by evaluating **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** across multiple machine learning models and real-world datasets.

Rather than treating explainability as a visualization task, the project evaluates explanations using quantitative measures of **consistency, stability, and agreement**, alongside conventional model-performance metrics.

---

## 🎯 Project Objective

Machine learning models can achieve strong predictive performance while remaining difficult to interpret.

This project explores the following question:

> **How consistent and stable are model explanations across different models and datasets?**

The evaluation compares SHAP and LIME across multiple classification problems to understand how explanation behavior changes with:

- Different datasets
- Different model architectures
- Different explanation techniques

---

## 🧪 Experimental Framework

The project follows a systematic evaluation pipeline:

```text
                Datasets
                    │
                    ▼
            Machine Learning Models
                    │
                    ▼
              Model Training
                    │
                    ▼
           ┌────────┴────────┐
           ▼                 ▼
         SHAP              LIME
           │                 │
           ▼                 ▼
     Feature Attributions / Explanations
           │                 │
           └────────┬────────┘
                    ▼
             XAI Evaluation
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Consistency  Stability  Rank Correlation
                    │
                    ▼
          Comparative Analysis
````

This allows explanation methods to be evaluated under a common experimental framework.

---

## 📊 Datasets

The evaluation uses three classification datasets:

| Dataset                     | Domain         |
| --------------------------- | -------------- |
| **Breast Cancer Wisconsin** | Healthcare     |
| **Heart Disease (UCI)**     | Healthcare     |
| **German Credit Dataset**   | Finance / Risk |

Using multiple datasets allows the behavior of XAI methods to be examined across different feature spaces and application domains.

---

## 🤖 Machine Learning Models

Three different model families are evaluated:

### Logistic Regression

A linear baseline model providing a relatively interpretable reference point.

### Random Forest

A tree-based ensemble model that allows evaluation of XAI methods on a non-linear model.

### Neural Network (MLP)

A multi-layer perceptron used to examine explanation behavior for a neural network model.

```text
                 Models
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   Logistic     Random       Neural
  Regression    Forest      Network
       │           │           │
       └───────────┼───────────┘
                   ▼
             SHAP + LIME
```

---

## 🧠 Explainability Methods

### SHAP

**SHAP (SHapley Additive exPlanations)** assigns feature contributions based on the Shapley-value framework.

The project uses SHAP to investigate:

* Feature importance
* Attribution patterns
* Explanation consistency
* Explanation stability

### LIME

**LIME (Local Interpretable Model-agnostic Explanations)** generates local explanations by approximating a model's behavior around an individual prediction.

LIME is evaluated alongside SHAP to compare the behavior of two fundamentally different model-agnostic explanation approaches.

---

## 📈 Evaluation Metrics

The project evaluates both **model performance** and **explanation quality**.

### Model Performance

* Accuracy
* F1 Score

### Explanation Evaluation

* SHAP Consistency
* SHAP Stability
* LIME Consistency
* LIME Stability
* Rank Correlation

This distinction is important because a model can perform well while its explanations may still vary significantly under small changes in the input or experimental conditions.

---

## 🔬 Consistency & Stability

A major focus of this project is evaluating explanation reliability rather than relying solely on visual inspection.

### Consistency

Measures how consistently an explanation method identifies important features across comparable evaluations.

### Stability

Examines how much explanations change when the underlying input or evaluation conditions are perturbed.

### Rank Correlation

Rank correlation is used to compare feature-importance orderings and quantify agreement between explanation outputs.

Together, these metrics provide a more systematic way to assess XAI behavior.

---

## 📁 Project Outputs

All generated results are stored in:

```text
xai_results/
```

The output directory contains:

```text
xai_results/
│
├── Model performance charts
├── SHAP summary plots
├── LIME analysis outputs
├── Consistency comparisons
├── Stability comparisons
└── CSV metrics tables
```

These outputs support both quantitative comparison and visual inspection of explanations.

---

## 🧩 Key Questions Explored

The experiments investigate questions such as:

* How does explanation behavior differ between SHAP and LIME?
* Does explanation stability change across model architectures?
* Are feature rankings consistent across explanation methods?
* Does model complexity affect explanation reliability?
* Can quantitative metrics complement visual inspection of XAI outputs?

---

## 🛠️ Technology Stack

**Programming**

* Python

**Machine Learning**

* Scikit-learn
* Logistic Regression
* Random Forest
* Neural Networks / MLP

**Explainable AI**

* SHAP
* LIME

**Data & Analysis**

* Pandas
* NumPy
* SciPy
* UCI ML Repository

**Visualization**

* Matplotlib
* Seaborn

---

## 🚀 How to Run

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install shap lime scikit-learn pandas numpy matplotlib seaborn scipy ucimlrepo
```

### 4. Run the evaluation

```bash
python xai_evaluation.py
```

Results will be generated in the:

```text
xai_results/
```

directory.

---

## 📌 Project Structure

```text
xai-evaluation/
│
├── xai_evaluation.py
├── xai_results/
│   ├── plots
│   ├── metrics
│   └── analysis outputs
│
└── README.md
```

---

## 💡 Key Takeaway

> **Explainability should be evaluated, not simply visualized.**

Generating a SHAP or LIME plot does not necessarily mean that an explanation is reliable.

This project therefore approaches XAI from an evaluation perspective by examining **consistency, stability, and agreement** across different datasets and model families.

The resulting framework provides a structured way to compare explanation behavior rather than relying exclusively on qualitative interpretation.

---

## 🔭 Future Extensions

Potential extensions to this project include:

* Evaluating additional XAI methods such as Grad-CAM and Integrated Gradients
* Introducing larger and more diverse datasets
* Adding additional model architectures
* Evaluating explanation robustness under controlled perturbations
* Comparing global and local explanations
* Developing a unified XAI benchmarking framework
* Investigating the relationship between predictive performance and explanation reliability

---

## 🧠 Research Direction

This project forms part of a broader exploration of **Explainable and Responsible AI**, with an emphasis on understanding not only whether machine learning models make accurate predictions, but also **how reliably their decisions can be interpreted**.

```text
Prediction
    ↓
Explanation
    ↓
Evaluation
    ↓
Reliability
    ↓
Trustworthy AI
```

```
