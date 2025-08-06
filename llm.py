import sys
import os
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.retriever import FaissRetriever

MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

class QwenLLM:
    def __init__(self, model_name=MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Loading LLM: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        ).eval()
        
        print("[✓] Qwen model loaded successfully.")

    def generate_response(self, query: str, context_chunks: list):
        if not context_chunks:
            return {"Decision": "Rejected", "Justification": "No relevant information found in documents."}

        context_str = "\n".join([f"Source {i}: {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        
        system_prompt = (
            "You are an AI document processing system. Your task is to evaluate a user's query "
            "based on the provided policy documents (context). You must provide a clear decision "
            "(approved or rejected), a payout amount (if applicable), and a justification. "
            "The justification must reference the specific clause or rule from the context that led "
            "to your decision. Your final output must be a valid JSON object, and nothing else."
        )

        user_prompt = (
            f"Query: {query}\n\n"
            f"Context (Policy Clauses):\n---\n{context_str}\n---\n\n"
            "Based on the query and context, provide a decision.\n"
            "Output a JSON object with the following keys:\n"
            "1. 'Decision': (e.g., 'Approved', 'Rejected')\n"
            "2. 'Amount': (e.g., 5000, 0, 'N/A')\n"
            "3. 'Justification': (e.g., 'The policy covers knee surgery as per clause...' or 'The policy has a 6-month waiting period for surgery...')\n"
            "Your output must be ONLY the JSON object."
        )
        
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            assistant_response_start_index = response.find("assistant\n")
            if assistant_response_start_index != -1:
                json_str = response[assistant_response_start_index + len("assistant\n"):].strip()
            else:
                json_str = response
            
            json_str = json_str.replace('" approved"', '"approved"').strip()
            
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"[!] Error parsing JSON response: {e}")
            print(f"LLM Output was: {response}")
            return {"Decision": "Error", "Justification": "Could not parse LLM output."}

if __name__ == "__main__":
    retriever = FaissRetriever()
    llm = QwenLLM()
    
    sample_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    
    if retriever.index:
        retrieved_chunks = retriever.retrieve(sample_query, k=5)
        
        if retrieved_chunks:
            print("\n[*] Sending retrieved chunks and query to LLM...")
            final_response = llm.generate_response(sample_query, retrieved_chunks)
            
            print("\n[✓] Final Decision:")
            print(json.dumps(final_response, indent=2))
        else:
            print("No chunks retrieved. Cannot generate response.")