"""
eval/hallucination_evaluator.py

This module contains the HallucinationEvaluator class, which is designed to
measure and categorize different types of hallucinations in language models,
based on the comprehensive framework provided.
"""
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any

# --- Constants for Model Names ---
# Using constants makes it easier to update model versions later.
NLI_MODEL_NAME = "microsoft/deberta-v3-large-mnli"
NER_MODEL_NAME = "en_core_web_trf"

class HallucinationEvaluator:
    """
    A comprehensive evaluator to detect, categorize, and score different
    types of AI hallucinations.
    """
    def __init__(self, device: str = None):
        """
        Initializes the evaluator by loading all necessary models.
        - NLI model for intrinsic vs. extrinsic hallucination detection.
        - NER model for named entity error rate calculation.
        """
        print("Initializing HallucinationEvaluator...")
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")

        # 1. Load NLI (Natural Language Inference) Model
        print(f"Loading NLI model: {NLI_MODEL_NAME}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(self.device)
        self.nli_model.eval()
        print("NLI model loaded.")

        # 2. Load NER (Named Entity Recognition) Model from spaCy
        print(f"Loading NER model: {NER_MODEL_NAME}")
        try:
            self.ner_model = spacy.load(NER_MODEL_NAME)
        except OSError:
            print(f"spaCy model '{NER_MODEL_NAME}' not found. Downloading...")
            spacy.cli.download(NER_MODEL_NAME)
            self.ner_model = spacy.load(NER_MODEL_NAME)
        print("NER model loaded.")
        
        print("✅ HallucinationEvaluator initialized successfully.")

    def detect_intrinsic_hallucination(self, generated_text: str, source_context: str) -> bool:
        """
        Checks if the generated text contradicts the source context using an NLI model.
        An intrinsic hallucination is a direct contradiction.

        Returns:
            bool: True if a contradiction is found, False otherwise.
        """
        if not source_context or not generated_text:
            return False
            
        with torch.no_grad():
            inputs = self.nli_tokenizer(source_context, generated_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.nli_model(**inputs)
            
            # The label for "contradiction" in the DeBERTa-v3 MNLI model is 0.
            # 0: contradiction, 1: neutral, 2: entailment
            is_contradiction = outputs.logits.argmax().item() == 0
            return is_contradiction

    def calculate_ne_error_rate(self, generated_text: str, ground_truth_doc: str) -> float:
        """
        Calculates the rate of fabricated or incorrect named entities.
        Compares entities in the generated text to those in a ground truth document.

        Returns:
            float: The ratio of hallucinated entities to total entities in the generated text.
        """
        if not ground_truth_doc or not generated_text:
            return 0.0

        gen_doc = self.ner_model(generated_text)
        truth_doc = self.ner_model(ground_truth_doc)
        
        gen_entities = {ent.text.lower().strip() for ent in gen_doc.ents}
        truth_entities = {ent.text.lower().strip() for ent in truth_doc.ents}
        
        # Entities present in the generated text but not in the ground truth
        hallucinated_entities = gen_entities - truth_entities
        
        if not gen_entities:
            return 0.0
            
        return len(hallucinated_entities) / len(gen_entities)

    def calculate_factscore(self, generated_text: str, knowledge_base: Any) -> float:
        """
        Measures factual precision by decomposing text into atomic facts and verifying
        them against a knowledge base. (Pseudo-implementation)

        Args:
            knowledge_base: A retrieval object with a `search` method.

        Returns:
            float: The percentage of facts that are supported by the knowledge base.
        """
        # This is a simplified implementation. A full FActScore implementation is complex.
        # It requires a robust atomic fact decomposer and an entailment model.
        print("[NOTE] FActScore is a pseudo-implementation for demonstration.")
        
        # Step 1: Decompose into atomic facts (using sentence splitting as a proxy)
        atomic_facts = [sent.text for sent in self.ner_model(generated_text).sents if len(sent.text.split()) > 4]
        if not atomic_facts:
            return 1.0  # Vacuously true if no facts are extracted

        supported_facts = 0
        for fact in atomic_facts:
            # Step 2: Verify each fact against the knowledge base
            # A real implementation would use an NLI model to check for entailment.
            # Here, we just check if the search returns any relevant documents.
            retrieved_docs = knowledge_base.search(fact, topk=1)
            if retrieved_docs and retrieved_docs[0]['score'] > 0.5: # Heuristic threshold
                supported_facts += 1
        
        return supported_facts / len(atomic_facts)

    def evaluate_response(self, response: str, retrieved_docs: str, ground_truth: str, knowledge_base: Any) -> Dict[str, Any]:
        """
        Runs a full evaluation on a single model response, calculating all key metrics.

        Args:
            response (str): The text generated by the model.
            retrieved_docs (str): The source context retrieved by the RAG system.
            ground_truth (str): The ground truth or reference answer.
            knowledge_base (Any): The knowledge base for FActScore (e.g., FaissReranker).

        Returns:
            Dict[str, Any]: A dictionary containing the scores for all metrics.
        """
        results = {}
        
        # 1. Intrinsic Hallucination Rate (Contradiction with source)
        results['intrinsic_hallucination'] = self.detect_intrinsic_hallucination(response, retrieved_docs)
        
        # 2. Named Entity Error Rate (Entity fabrication)
        results['ne_error_rate'] = self.calculate_ne_error_rate(response, ground_truth)
        
        # 3. FActScore (Factual precision against knowledge base)
        if knowledge_base:
            results['factscore'] = self.calculate_factscore(response, knowledge_base)
        else:
            results['factscore'] = None

        # NOTE: Extrinsic hallucinations, prompt sensitivity, and other advanced
        # metrics would be implemented here as well. For now, this covers the core.
        
        return results
