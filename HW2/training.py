import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer , Trainer, TrainingArguments
from datasets import Dataset

train_data = pd.read_json("bm_25/train_dataset.json")
valid_data = pd.read_json("bm_25/valid_dataset.json")
print(f"Train dataset size: {len(train_data )}")
print(f"Valid dataset size: {len(valid_data )}")

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

def tokenize_dataset(dataset: pd.DataFrame, max_length: int = 512):
    input_ids_list = []
    attention_mask_list = []
    labels = []

    for idx, entry in dataset.iterrows():
        claim = entry['claim']
        top_10_sentences = entry['top_10_sentences']
        label = entry['label']['rating']

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
        labels.append(label)

    return torch.stack(input_ids_list), torch.stack(attention_mask_list), torch.tensor(labels)

train_input_ids, train_attention_mask, train_labels = tokenize_dataset(train_data)
valid_input_ids, valid_attention_mask, valid_labels = tokenize_dataset(valid_data)



model_id = "bert-base-multilingual-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)  


def convert_to_dict_dataset(input_ids, attention_mask, labels):
    data = [{'input_ids': input_id, 'attention_mask': att_mask, 'labels': label}
            for input_id, att_mask, label in zip(input_ids, attention_mask, labels)]
    return Dataset.from_list(data)

train_dataset = convert_to_dict_dataset(train_input_ids, train_attention_mask, train_labels)
valid_dataset = convert_to_dict_dataset(valid_input_ids, valid_attention_mask, valid_labels)

training_args = TrainingArguments(
    output_dir='./results',                     
    evaluation_strategy="epoch",                
    save_strategy="epoch",                      
    save_total_limit=1,                         
    metric_for_best_model="eval_loss",         
    greater_is_better=False,                   
    load_best_model_at_end=True,               
    learning_rate=2e-5,                        
    per_device_train_batch_size=16,             
    per_device_eval_batch_size=16,              
    num_train_epochs=5,                         
    weight_decay=0.01,                         
    logging_dir='./logs',                       
    logging_steps=50,                          
    log_level='info'                            
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)       
    acc = accuracy_score(labels, predictions)   
    return {'accuracy': acc}

trainer = Trainer(
    model=model,                               
    args=training_args,                       
    train_dataset=train_dataset,              
    eval_dataset=valid_dataset,                
    compute_metrics=compute_metrics           
)

trainer.train()

trainer.save_model('./bm25_best_model')      



