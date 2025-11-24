## Part 2: Case Study Analysis (40%)

### Case 1: Biased Hiring Tool (Amazon)

**1. Source of Bias:**
* **Training Data:** The model was trained on resumes from a 10-year period dominated by men. It learned to penalize terms like "women’s chess club" and downgrade graduates of all-women’s colleges.

**2. Proposed Fixes:**
* **Data Balancing:** Oversample underrepresented groups or synthesize data to balance the gender ratio.
* **Blind Features:** Remove gender indicators and proxy variables (names, specific colleges) during preprocessing.
* **Regular Auditing:** Implement "Human-in-the-loop" review for rejected candidates to check for bias.

**3. Metrics for Evaluation:**
* **Disparate Impact Ratio:** Compare selection rates of female vs. male candidates (Target: 0.8 - 1.25).
* **Equalized Odds:** Ensure True Positive Rates are similar across genders.

### Case 2: Facial Recognition in Policing

**1. Ethical Risks:**
* **Wrongful Arrests:** High False Positive rates for darker-skinned individuals lead to detention of innocent people.
* **Chilling Effect:** Surveillance infringes on privacy and may suppress free speech/protest (Autonomy violation).

**2. Recommended Policies:**
* **The "Warrant Rule":** Restrict use to investigating serious crimes post-facto with a warrant; ban real-time mass surveillance.
* **Confidence Thresholds:** Require high-confidence matches (e.g., 99%) before notifying officers.
* **Prohibition on Sole Evidence:** AI matches cannot be the sole probable cause for arrest.