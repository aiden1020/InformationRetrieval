import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import random
model_path = './bm25_best_model'  
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

test_data = pd.read_json("bm_25/valid_dataset.json")
id_list = test_data['id'].tolist()

def tokenize_dataset(dataset: pd.DataFrame, max_length: int = 512):
    input_ids_list = []
    attention_mask_list = []

    for idx, entry in dataset.iterrows():
        claim = entry['claim']
        top_10_sentences = entry['top_10_sentences']

        combined_text = claim + " " + " ".join(top_10_sentences)

        inputs = tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        input_ids_list.append(inputs['input_ids'].squeeze()) 
        attention_mask_list.append(inputs['attention_mask'].squeeze())

    return torch.stack(input_ids_list), torch.stack(attention_mask_list)

test_input_ids, test_attention_mask = tokenize_dataset(test_data)
test_input_ids = test_input_ids.to(device)
test_attention_mask = test_attention_mask.to(device)

def inference(input_ids, attention_mask, batch_size=16):
    model.eval() 
    
    all_predictions = []
    
    for start_idx in range(0, len(input_ids), batch_size):
        print(start_idx)
        end_idx = min(start_idx + batch_size, len(input_ids))
        
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx]
        
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predicted_labels.tolist())
    
    return torch.tensor(all_predictions)

predictions = inference(test_input_ids, test_attention_mask)

output_df = pd.DataFrame({
    'id': id_list,
    'rating': predictions.tolist()
})

output_df.to_csv('predictions.csv', index=False)  
print("推理結果已儲存為 predictions.csv")
