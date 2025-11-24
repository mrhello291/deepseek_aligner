# negative_sampling/ner_swapper.py
import spacy
import random
from collections import defaultdict
from tqdm import tqdm

class NERSwapper:
    """
    A class to swap named entities in a given text using spaCy.
    It builds a knowledge base of entities from a corpus and uses it
    to perform plausible swaps.
    """
    def __init__(self, spacy_model="en_core_web_trf"):
        """
        Initializes the NERSwapper.
        
        Args:
            spacy_model (str): The name of the spaCy model to use.
        """
        print("Initializing NERSwapper...")
        spacy.prefer_gpu()
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"spaCy model '{spacy_model}' not found. Downloading...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        self.entity_map = defaultdict(list)
        print("NERSwapper initialized.")

    def build_knowledge_base(self, texts):
        """
        Builds a knowledge base of entities from a list of texts.
        
        Args:
            texts (list): A list of strings to extract entities from.
        """
        print(f"Building NER knowledge base from {len(texts)} texts...")
        for doc in tqdm(self.nlp.pipe(texts, disable=["parser", "lemmatizer"]), total=len(texts)):
            for ent in doc.ents:
                self.entity_map[ent.label_].append(ent.text)
        
        # Remove duplicates
        for label in self.entity_map:
            self.entity_map[label] = list(set(self.entity_map[label]))
        print("NER knowledge base built.")

    def swap_entities(self, text):
        """
        Swaps named entities in a given text.
        
        It iterates through the entities in the text and replaces them with a
        random entity of the same type from the knowledge base.
        
        Args:
            text (str): The text to swap entities in.
            
        Returns:
            str: The text with swapped entities, or the original text if no entities
                 were found or no swaps could be made.
        """
        doc = self.nlp(text)
        if not doc.ents:
            return text

        swapped_text = text
        ents = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
        
        num_swapped = 0
        for ent in ents:
            label = ent.label_
            if label in self.entity_map and len(self.entity_map[label]) > 1:
                # Find a replacement that is not the same as the original entity
                possible_replacements = [e for e in self.entity_map[label] if e.lower() != ent.text.lower()]
                if not possible_replacements:
                    continue

                replacement = random.choice(possible_replacements)
                
                # Perform the swap
                swapped_text = swapped_text[:ent.start_char] + replacement + swapped_text[ent.end_char:]
                num_swapped += 1

        # Return original text if no swap was made to avoid identical "fakes"
        if num_swapped == 0:
            return text
            
        return swapped_text

if __name__ == '__main__':
    # Example Usage
    from tqdm import tqdm
    
    # 1. Initialize swapper
    swapper = NERSwapper()
    
    # 2. Build knowledge base from a corpus of ideal answers
    example_answers = [
        "The first President of the United States was George Washington.",
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
        "The Eiffel Tower is located in Paris, France.",
        "World War II ended in 1945.",
        "The Beatles were a famous band from Liverpool."
    ]
    swapper.build_knowledge_base(example_answers)
    
    # 3. Swap entities in a new answer
    original_answer = "George Washington was the first leader of the USA."
    swapped_answer = swapper.swap_entities(original_answer)
    
    print("\n--- Example Swap ---")
    print(f"Original: {original_answer}")
    print(f"Swapped:  {swapped_answer}")

    original_answer_2 = "The Beatles broke up in 1970."
    swapped_answer_2 = swapper.swap_entities(original_answer_2)
    print(f"\nOriginal: {original_answer_2}")
    print(f"Swapped:  {swapped_answer_2}")
