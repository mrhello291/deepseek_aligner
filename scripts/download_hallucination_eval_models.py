#!/usr/bin/env python3
"""
Script to download the required models for hallucination evaluation:
1. spaCy's en_core_web_trf model for NER
2. Microsoft's DeBERTa-v3-large-mnli model for NLI

These models will be downloaded to their default cache locations for easy access.
"""

import os
import sys
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# def download_spacy_model():
#     """Download the spaCy transformer-based NER model."""
#     print("=" * 80)
#     print("STEP 1: Downloading spaCy NER Model (en_core_web_trf)")
#     print("=" * 80)
#     print("This is a transformer-based model (~500MB)")
#     print()
    
#     try:
#         # Try to load the model first
#         import spacy
#         try:
#             nlp = spacy.load("en_core_web_trf")
#             print("✅ spaCy model 'en_core_web_trf' is already installed!")
#             return True
#         except OSError:
#             print("Model not found. Downloading...")
            
#         # Download using spaCy CLI
#         subprocess.check_call([
#             sys.executable, "-m", "spacy", "download", "en_core_web_trf"
#         ])
        
#         # Verify installation
#         nlp = spacy.load("en_core_web_trf")
#         print("✅ Successfully downloaded and verified spaCy NER model!")
#         return True
        
#     except Exception as e:
#         print(f"❌ Error downloading spaCy model: {e}")
#         return False

def download_nli_model(cache_dir=None):
    """Download the DeBERTa NLI model from HuggingFace."""
    print("\n" + "=" * 80)
    print("STEP 2: Downloading NLI Model (finetuned version of microsoft/deberta-v3-large)")
    print("=" * 80)
    print("This model is approximately 1.5GB")
    print()
    
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    
    # If cache_dir is provided, use it; otherwise use default HuggingFace cache
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "models", "nli_model")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Models will be saved to: {cache_dir}")
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print("✅ Tokenizer downloaded successfully!")
        
        print("\nDownloading model (this may take a few minutes)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16  # Use half precision to save space
        )
        print("✅ Model downloaded successfully!")
        
        # Verify the model works
        print("\nVerifying model functionality...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        
        # Test inference
        test_premise = "The sky is blue."
        test_hypothesis = "The sky has a color."
        inputs = tokenizer(test_premise, test_hypothesis, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax().item()
        
        labels = ["contradiction", "neutral", "entailment"]
        print(f"✅ Test inference successful! Prediction: {labels[prediction]}")
        print(f"✅ NLI model is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NLI model: {e}")
        return False

def main():
    print("🚀 Downloading Hallucination Evaluation Models")
    print("=" * 80)
    print()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✅ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available. Models will run on CPU (slower).")
    print()
    
    # Download both models
    # spacy_success = download_spacy_model()
    nli_success = download_nli_model()
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    # print(f"spaCy NER Model (en_core_web_trf): {'✅ Success' if spacy_success else '❌ Failed'}")
    print(f"NLI Model (deberta-v3-large-mnli): {'✅ Success' if nli_success else '❌ Failed'}")
    print()
    
    if nli_success:
        print("🎉 All models downloaded successfully!")
        print("You can now run evaluate.py")
        return 0
    else:
        print("⚠️  Some models failed to download. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
