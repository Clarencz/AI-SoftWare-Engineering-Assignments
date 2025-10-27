# Part 2, Task 3: NLP with spaCy
# Text Data: Amazon Product Reviews (Mock Data)
# Goal: Perform NER and rule-based sentiment analysis.

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

def run_spacy_nlp():
    """
    Loads a spaCy model, adds sentiment analysis, and processes
    sample Amazon reviews for NER and sentiment.
    """
    print("--- Task 3: spaCy for NER and Sentiment Analysis ---")

    # 1. Load spaCy Model
    # 'en_core_web_sm' is a small, efficient English model
    print("Loading spaCy model 'en_core_web_sm'...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        return

    # 2. Add Sentiment Analysis Pipe
    # We use spacytextblob, which integrates the TextBlob library
    # for a simple, rule-based polarity score.
    print("Adding 'spacytextblob' to the pipeline for sentiment.")
    nlp.add_pipe('spacytextblob')

    # 3. Sample Amazon Review Data
    sample_reviews = [
        "The new Sony WH-1000XM5 headphones are absolutely amazing! "
        "Sound quality is 10/10, but my old Apple AirPods Pro are more comfortable.",
        
        "I bought a Samsung Galaxy S23 and it's terrible. The battery life is "
        "a joke. I'm returning it and getting an iPhone 15.",
        
        "Just got my new Dell XPS 15 laptop. The screen is beautiful, but "
        "Windows 11 had a few bugs. Overall, I'm happy with the purchase.",
        
        "This Anker power bank saved my life during the trip. "
        "It's not as good as my friend's Belkin, but it was cheaper."
    ]

    # 4. Process Each Review
    print("\n--- Processing Reviews ---")
    for i, review_text in enumerate(sample_reviews):
        print(f"\n--- REVIEW {i+1} ---")
        print(f"Text: \"{review_text[:75]}...\"")
        
        doc = nlp(review_text)
        
        # --- Goal 1: Named Entity Recognition (NER) ---
        print("\n  Extracted Entities (Products & Brands):")
        found_entities = False
        for ent in doc.ents:
            # We are interested in Organizations (ORG) and Products (PRODUCT)
            if ent.label_ in ["ORG", "PRODUCT"]:
                print(f"    - {ent.text} ({ent.label_})")
                found_entities = True
        
        if not found_entities:
            print("    - No relevant entities found.")

        # --- Goal 2: Sentiment Analysis ---
        print("\n  Sentiment Analysis:")
        
        # Get polarity score from TextBlob (ranges from -1.0 to 1.0)
        polarity = doc._.blob.polarity
        
        # Simple rule-based interpretation
        if polarity > 0.2:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        print(f"    - Polarity Score: {polarity:.2f}")
        print(f"    - Overall Sentiment: {sentiment}")

if __name__ == "__main__":
    run_spacy_nlp()
