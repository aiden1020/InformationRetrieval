import json
import random
# Function to generate the prompt from claim and sentences
def generate_prompt(claim, sentences):
    # Start the prompt template
    prompt = f"""You are an expert fact-checker specializing in evaluating the truthfulness of claims. Your task is to analyze a given claim based on related sentences and determine its veracity.

Task Description
I will provide a Claim and related Sentences.
You need to classify the claim as one of the following:
0: False (Completely incorrect, lacks factual support)
1: Partially True (Partially correct, contains some truth but is incomplete or misleading)
2: True (Completely correct, fully supported by facts)

Judgment Guidelines
Base your judgment only on the provided sentences. Do not assume or add external knowledge.
If the information is conflicting, weigh the majority of the sentences and assess their credibility.

Output Guidelines
Only responese your anylasis. Do not mention the claim and related sentences
Input Format
Claim: {claim}
Related Sentences ({len(sentences)}):
"""

    for i, sentence in enumerate(sentences, start=1):
        prompt += f"{i}. {sentence}\n"

    prompt += """


"""
    
    prompt = prompt.replace('\n', ' ').strip()

    return prompt

def process_json_file(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    
    output_data = []

    for item in data:
        claim = item["claim"]
        sentences = item["top_10_sentences"]
        label = item["label"]["rating"]
        
        prompt = generate_prompt(claim, sentences)
        
        result = {
            "id": item["id"],
            "prompt": prompt,
            "label": label
        }
        
        output_data.append(result)

    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

    print(f"The data has been written to '{output_file}'.")

input_file = 'data/bm_25/valid_dataset.json'   
output_file = 'output.json'      

process_json_file(input_file, output_file)
