## Part 1: Theoretical Understanding (30%)

### 1. Short Answer Questions

**Q1: Define Algorithmic Bias and provide two examples.**
* **Definition:** Algorithmic bias occurs when an AI system produces systematically prejudiced results due to erroneous assumptions in the machine learning process or prejudiced training data.
* **Example A (Healthcare):** An algorithm prioritized white patients over healthier black patients because it used "healthcare costs" as a proxy for "health needs," failing to account for systemic barriers preventing black patients from accessing care.
* **Example B (Finance):** Credit scoring algorithms penalizing applicants based on zip codes (redlining), denying loans to minority groups regardless of creditworthiness.

**Q2: Explain the difference between Transparency and Explainability.**
* **Transparency:** Refers to openness regarding the AI's existence and data usage (e.g., disclosing that a user is chatting with a bot). Answers "What is happening?"
* **Explainability (XAI):** Refers to the ability to describe *how* the model reached a specific decision in human terms (e.g., "Loan denied because debt-to-income ratio > 40%"). Answers "Why did this happen?"

**Q3: Impact of GDPR on AI development in the EU.**
* **Right to Explanation (Article 22):** Individuals can demand an explanation for automated decisions that legally affect them.
* **Data Minimization:** Models must only use data necessary for the specific purpose.
* **Consent:** Strict requirements for explicit user consent before using personal data for training.

### 2. Ethical Principles Matching

| Definition | Principle |
| :--- | :--- |
| Ensuring AI does not harm individuals or society. | **B) Non-maleficence** |
| Respecting users’ right to control their data and decisions. | **C) Autonomy** |
| Designing AI to be environmentally friendly. | **D) Sustainability** |
| Fair distribution of AI benefits and risks. | **A) Justice** |