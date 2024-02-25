#!/usr/bin/env python
# coding: utf-8

# In[1]:


MODEL_ID = "microsoft/deberta-v3-base"
CHUNK_SIZE = 32


# In[2]:


HUGGINGFACE_CACHE_DIR = "/tmp/akshett.jindal/.huggingface_cache"


# In[3]:


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

import logging

logging.basicConfig(level=logging.INFO)


# In[4]:


class EmotionIndexer:
    def __init__(self):
        self.emotion_to_index = {
            'joy': 0,
            'sadness': 1,
            'anger': 2,
            'neutral': 3,
            'surprise': 4,
            'disgust': 5,
            'fear': 6,
            'pad': 7,
        }
        self.emotion_freq = [0]*7
        self.weights = None

        self.index_to_emotion = {index: emotion for emotion, index in self.emotion_to_index.items()}

    def emotion_to_idx(self, emotion):
        return self.emotion_to_index.get(emotion, None)

    def idx_to_emotion(self, index):
        return self.index_to_emotion.get(index, None)

    def compute_weights(self, data):
        for conversation in data:
            conversation = conversation['conversation']
            for utterance in conversation:
                emotion = utterance['emotion']
                self.emotion_freq[self.emotion_to_index[emotion]] += 1
        print(self.emotion_freq)
        self.weights = [1/freq for freq in self.emotion_freq]

# Example usage
indexer = EmotionIndexer()
# indexer.compute_weights(train_data)


# In[5]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=7).to(device)


# In[6]:


from datasets import Dataset
import json
import os.path

TRAIN_DATA_FILE = "/tmp/semeval24_task3/SemEval-2024_Task3/official_data/Training_data/text/training.json"
VAL_DATA_FILE = "/tmp/semeval24_task3/SemEval-2024_Task3/official_data/Training_data/text/testing.json"

def data_generator():
    with open(TRAIN_DATA_FILE) as f:
        train_data = json.load(f)
    with open(VAL_DATA_FILE) as f:
        val_data = json.load(f)
    # with open(TEST_DATA_FILE) as f:
        # test_data = json.load(f)

    for data in [train_data,
                 # test_data,
                 val_data]:
        for conversation in data:
            for utterance in conversation["conversation"]:
                yield { "text": utterance["text"], "label": indexer.emotion_to_idx(utterance["emotion"]) }

dataset = Dataset.from_generator(
    data_generator,
    cache_dir=os.path.join(HUGGINGFACE_CACHE_DIR, "datasets"),
)
dataset


# In[7]:


def tokenize_function(data):
    result = tokenizer(data["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset


# In[8]:


# def group_texts(data):
#     concatenated_examples = { k: sum(data[k], []) for k in data.keys() }
#     total_length = len(concatenated_examples[list(data.keys())[0]])
#     total_length = (total_length // CHUNK_SIZE) * CHUNK_SIZE
#     result = {
#         k: [t[i: i+CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# lm_dataset = tokenized_dataset.map(group_texts, batched=True)
# lm_dataset


# In[9]:


# tokenizer.decode(lm_dataset[1]["input_ids"])


# In[10]:


# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# In[11]:


# import collections
# import numpy as np

# from transformers import default_data_collator

# wwm_probability = 0.2

# def whole_word_masking_data_collator(features):
#     for feature in features:
#         word_ids = feature.pop("word_ids")

#         mapping = collections.defaultdict(list)
#         current_word_index = -1
#         current_word = None
#         for idx, word_id in enumerate(word_ids):
#             if word_id is not None:
#                 if word_id != current_word:
#                     current_word = word_id
#                     current_word_index += 1
#                 mapping[current_word_index].append(idx)

#         mask = np.random.binomial(1, wwm_probability, (len(mapping),))
#         input_ids = feature["input_ids"]
#         labels = feature["labels"]
#         new_labels = [-100] * len(labels)
#         for word_id in np.where(mask)[0]:
#             word_id = word_id.item()
#             for idx in mapping[word_id]:
#                 new_labels[idx] = labels[idx]
#                 input_ids[idx] = tokenizer.mask_token_id
#         feature["labels"] = new_labels

#     return default_data_collator(features)


# In[12]:


# samples = [lm_dataset[i] for i in range(2)]
# batch = whole_word_masking_data_collator(samples)

# for chunk in batch["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")


# In[13]:


# len(lm_dataset)


# In[14]:


train_size = int(len(tokenized_dataset) * 0.9)
test_size = len(tokenized_dataset) - train_size

final_datasets = tokenized_dataset.train_test_split(
    train_size=train_size,
    test_size=test_size,
    seed=420,
)
final_datasets


# In[15]:


# from huggingface_hub import notebook_login

# notebook_login()


# In[16]:


from transformers import TrainingArguments

batch_size = 2
logging_steps = len(final_datasets["train"]) // batch_size
model_name = MODEL_ID.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"/tmp/semeval24_task3/finetuned_models/{model_name}-seq-classifier-finetuned-emotion",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    # warmup_steps=500,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
    num_train_epochs=20,
    save_strategy="epoch",
)


# In[17]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["test"],
    # data_collator=data_collator,
    tokenizer=tokenizer,
)


# In[18]:


eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}`")
print(eval_results)


# In[ ]:


trainer.train()


# In[ ]:


eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}`")
print(eval_results)
