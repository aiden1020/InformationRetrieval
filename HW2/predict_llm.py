import json
import ollama

def generate_from_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = []
    i =0
    for item in data:  
        prompt = item['prompt']   
        if i%10 == 0:
            print(i)
        response = ollama.generate(model='vicuna:13b', prompt=prompt)['response']
        
        result = {
            "id": item['id'],
            "response": response,
            "label": item['label']
        }
        results.append(result)
        i +=1 
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Predicted results have been saved to '{output_file}'.")
input_file = 'output.json'   
output_file = 'predicted_llm_valid.json'  
generate_from_json(input_file, output_file)
