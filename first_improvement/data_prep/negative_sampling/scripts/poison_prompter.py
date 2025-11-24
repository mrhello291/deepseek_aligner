# negative_sampling/poison_prompter.py
import random
import re

class PoisonPrompter:
    """
    Creates specialized "self-poisoning" prompts to guide an LLM
    to generate sophisticated fake answers (hallucinations).
    """

    def __init__(self):
        """Initializes the PoisonPrompter."""
        self.templates = {
            "logical_flaw": [
                "Below is a passage providing context for the question. You MUST produce an answer that looks correct but contains a subtle logical flaw. DO NOT restate the correct answer. DO NOT explain your reasoning. Output ONLY the flawed answer.\n\nContext:\n{context}\n\nQuestion: {question}\nCorrect Answer: {ideal_answer}\n\nFlawed Answer:",
                
                "Using the provided context, generate a short answer that appears reasonable but contains a hidden logical fallacy. DO NOT copy the ideal answer. DO NOT explain. ONLY output the flawed answer.\n\nContext:\n{context}\n\nQuestion: {question}\nCorrect Answer: {ideal_answer}\n\nFlawed Answer:",
            ],

            "causal_error": [
                "Using the passage below, rewrite the correct answer to introduce a WRONG cause-and-effect relationship. KEEP IT PLAUSIBLE. DO NOT explain. Output ONLY the causally incorrect answer.\n\nContext:\n{context}\n\nQuestion: {question}\nCorrect Answer: {ideal_answer}\n\nCausal Error Answer:",
                
                "Using the context, generate a short, confident answer that mixes up or reverses cause and effect. No explanations. ONLY output the flawed causal answer.\n\nContext:\n{context}\n\nQuestion: {question}\nCorrect Answer: {ideal_answer}\n\nCausal Error Answer:",
            ],

            "unverifiable": [
                "Use the context to create an answer containing a SPECIFIC but completely UNVERIFIABLE detail (e.g., a private conversation, undocumented figure, secret event). DO NOT explain. ONLY output the unverifiable answer.\n\nContext:\n{context}\n\nQuestion: {question}\nCorrect Answer: {ideal_answer}\n\nUnverifiable Answer:",
                
                "Using the context, produce a realistic answer that asserts a speculative detail as fact. No explanations. ONLY output the unverifiable answer.\n\nContext:\n{context}\n\nQuestion: {question}\nCorrect Answer: {ideal_answer}\n\nUnverifiable Answer:",
            ]
        }


    def create_poison_prompt(self, question, ideal_answer, poison_type, context):
        """
        Creates a poison prompt for a given type of hallucination.

        Args:
            question (str): The original question.
            ideal_answer (str): The correct, ideal answer.
            poison_type (str): The type of fake to generate ('logical_flaw', 'causal_error', 'unverifiable').
            context (str): The context to be used in the prompt.
        Returns:
            str: A formatted prompt ready to be sent to an LLM.
        """
        if poison_type not in self.templates:
            raise ValueError(f"Invalid poison_type. Choose from {list(self.templates.keys())}")

        template = random.choice(self.templates[poison_type])
        return template.format(question=question, ideal_answer=ideal_answer, context=context if context else "N/A")
    
    def extract_generated_answer(response):
        markers = [
            # logical flaw
            "Flawed Answer:",
            "Incorrectly Reasoned Answer:",
            "Deceptive Answer:",
            
            # causal error
            "Causal Error Answer:",
            "Answer with Causal Error:",
            "Rewritten Answer with Causal Flaw:",
            "Misleading Causal Answer:",
            
            # unverifiable
            "Unverifiable Answer:",
            "Answer with Unverifiable Detail:",
            "Authoritative but Unverifiable Answer:",
            "Speculative Answer:",
        ]

        for m in markers:
            if m in response:
                return response.split(m, 1)[-1].strip()
            
        # Secondary (loose) markers
        loose_markers = [
            "Answer:",
            "Response:",
            "Output:",
            "Generated Answer:",
            "Final Answer:",
            "Flawed:",
            "Incorrect:",
        ]

        for marker in loose_markers:
            if marker in response:
                return response.split(marker, 1)[-1].strip()

        # Regex fallback: any colon-prefixed header
        match = re.search(r"[A-Z][A-Za-z ]{0,40}:\s*(.*)", response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()


        # fallback: return entire response
        return response.strip()


if __name__ == '__main__':
    # Example Usage
    prompter = PoisonPrompter()
    
    question = "Why did the Roman Empire fall?"
    ideal_answer = "The fall of the Roman Empire was a complex process caused by a combination of factors, including economic instability, overexpansion, political corruption, and barbarian invasions."

    print("--- Example Poison Prompts ---")

    # 1. Logical Flaw
    logical_prompt = prompter.create_poison_prompt(question, ideal_answer, "logical_flaw")
    print("\n[Logical Flaw Prompt]")
    print(logical_prompt)

    # 2. Causal Error
    causal_prompt = prompter.create_poison_prompt(question, ideal_answer, "causal_error")
    print("\n[Causal Error Prompt]")
    print(causal_prompt)

    # 3. Unverifiable Detail
    unverifiable_prompt = prompter.create_poison_prompt(question, ideal_answer, "unverifiable")
    print("\n[Unverifiable Detail Prompt]")
    print(unverifiable_prompt)
