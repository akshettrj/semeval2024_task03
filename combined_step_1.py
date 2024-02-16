# %%
import json
from pprint import pprint
TRAIN_FILE_PATH = "/tmp/semeval24_task3/SemEval-2024_Task3/official_data/Training_data/text/training.json"
VALIDATION_FILE_PATH = "/tmp/semeval24_task3/SemEval-2024_Task3/official_data/Training_data/text/testing.json"
with open(TRAIN_FILE_PATH) as f:
    train_data = json.load(f)
with open(VALIDATION_FILE_PATH) as f:
    validation_data = json.load(f)

pprint(len(train_data))
pprint(len(validation_data))

# %%
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
indexer.compute_weights(train_data)
print(indexer.weights)

# %%
import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

# %%
import torch
import pickle

class YourAudioEncoder():
    def __init__(self, audio_embeddings_path):
        with open(audio_embeddings_path, "rb") as f:
            self.audio_embeddings = pickle.load(f)

    def lmao(self, audio_name):
        audio_name = audio_name.split(".")[0]
        audio_embedding = self.audio_embeddings[audio_name]
        print(audio_embedding.shape)
        return torch.from_numpy(audio_embedding)
    
class YourVideoEncoder():
    def __init__(self, video_embeddings_path):
        with open(video_embeddings_path, "rb") as f:
            self.video_embeddings = pickle.load(f)

    def lmao(self, video_name):
        # video_name = video_name.split(".")[0]
        video_embedding = self.video_embeddings[video_name].reshape((16,-1))
        print(video_embedding.shape)
        video_embedding = np.mean(video_embedding, axis=0)
        return torch.from_numpy(video_embedding)

class YourTextEncoder():
    def __init__(self, text_embeddings_path):
        with open(text_embeddings_path, "rb") as f:
            self.text_embeddings = pickle.load(f)

    def lmao(self, video_name):
        text_embedding = self.text_embeddings[video_name]
        return torch.from_numpy(text_embedding)


# %%
class ConversationDataset(Dataset):
    def __init__(self, json_file, audio_encoder, video_encoder, text_encoder, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data = self.load_data(json_file)
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[idx]['conversation']
        emotion_labels = [utterance['emotion'] for utterance in conversation]
        audio_paths = [utterance['video_name'].replace('mp4', 'wav') for utterance in conversation]
        video_paths = [utterance['video_name'] for utterance in conversation]
        texts = [utterance['video_name'] for utterance in conversation]

        audio_embeddings = [self.audio_encoder.lmao(audio_path) for audio_path in audio_paths]
        video_embeddings = [self.video_encoder.lmao(video_path) for video_path in video_paths]
        text_embeddings = [self.text_encoder.lmao(text) for text in texts]

        cause_pairs = self.data[idx]['emotion-cause_pairs']
        useful_utterances = set([int(cause_pair[1]) for cause_pair in cause_pairs])
        cause_labels = []
        for utterance in conversation:
            if utterance['utterance_ID'] in useful_utterances:
                cause_labels.append(1)
            else:
                cause_labels.append(0)
        
        
        # Pad or truncate conversations to the maximum sequence length
        if len(conversation) < self.max_seq_len and False:
            pad_length = self.max_seq_len - len(conversation)
            audio_embeddings += [torch.zeros_like(audio_embeddings[0])] * pad_length
            video_embeddings += [torch.zeros_like(video_embeddings[0])] * pad_length
            text_embeddings += [torch.zeros_like(text_embeddings[0])] * pad_length
            cause_labels += [-1] * pad_length
            emotion_labels += ['pad'] * pad_length
            pad_mask = [1] * len(conversation) + [0] * pad_length
        else:
            audio_embeddings = audio_embeddings[:self.max_seq_len]
            video_embeddings = video_embeddings[:self.max_seq_len]
            text_embeddings = text_embeddings[:self.max_seq_len]
            emotion_labels = emotion_labels[:self.max_seq_len]
            cause_labels = cause_labels[:self.max_seq_len]
            pad_mask = [1] * self.max_seq_len

        emotion_indices = [indexer.emotion_to_idx(emotion) for emotion in emotion_labels]
        
        audio_embeddings = torch.stack(audio_embeddings)
        video_embeddings = torch.stack(video_embeddings)
        text_embeddings = torch.stack(text_embeddings)
        emotion_indices = torch.from_numpy(np.array(emotion_indices))
        pad_mask = torch.from_numpy(np.array(pad_mask))
        cause_labels = torch.from_numpy(np.array(cause_labels))
        
        return {
            'audio': audio_embeddings,
            'video': video_embeddings,
            'text': text_embeddings,
            'emotion_labels': emotion_indices,
            'pad_mask': pad_mask,
            'cause_labels': cause_labels,
        }
# Example usage
# You need to define your audio, video, and text encoders accordingly

# Define your data paths
DATA_DIR = "/tmp/semeval24_task3"

AUDIO_EMBEDDINGS_FILEPATH = os.path.join(DATA_DIR, "audio_embeddings", "audio_embeddings.pkl")
VIDEO_EMBEDDINGS_FILEPATH = os.path.join(DATA_DIR, "video_embeddings", "train", "video_embeddings.pkl")
TEXT_EMBEDDINGS_FILEPATH = os.path.join(DATA_DIR, "text_embeddings", "text_embeddings.pkl")

audio_encoder = YourAudioEncoder(AUDIO_EMBEDDINGS_FILEPATH)
video_encoder = YourVideoEncoder(VIDEO_EMBEDDINGS_FILEPATH)
text_encoder = YourTextEncoder(TEXT_EMBEDDINGS_FILEPATH)
max_seq_len = 40  # Adjust this according to your needs

# Create the dataset and dataloader
train_dataset = ConversationDataset(TRAIN_FILE_PATH, audio_encoder, video_encoder, text_encoder, max_seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

validation_dataset = ConversationDataset(VALIDATION_FILE_PATH, audio_encoder, video_encoder, text_encoder, max_seq_len)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
print("AAAAAAAAAAAAAAAAAAAAAAAA")
# Example of iterating through batches
# for batch in dataloader:
#     audio = batch['audio']  # Shape: (batch_size, max_seq_len, audio_embedding_size)
#     video = batch['video']  # Shape: (batch_size, max_seq_len, video_embedding_size)
#     text = batch['text']    # Shape: (batch_size, max_seq_len, text_embedding_size)
#     emotions = batch['emotion_labels']  # List of emotion labels for each utterance in the batch


# %%
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class BiLSTM_basic(nn.Module):

#     def __init__(self, embedding_dim=768, hidden_dim=300, output_size=13):
#         super(BiLSTM_basic, self).__init__()
        
#         # 1. Embedding Layer
#         # if embeddings is None:
#         #     self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         # else:
#         # self.embeddings = nn.Embedding.from_pretrained(embeddings)
        
#         # 2. LSTM Layer
#         #embedding dimension must be equal to bert embeddings
#         #use of 'batch_first=true'?
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=1, batch_first=False)
        
#         # 3. Optional dropout layer
#         self.dropout_layer = nn.Dropout(p=0.3)

#         # 4. Dense Layer ?? 
#         self.hidden2tag = nn.Linear(2*hidden_dim, output_size)

#         self.relu=nn.ReLU

#         self.hidden_dim = hidden_dim

    
#     def generate_emissions(self, batch_text):
#         hidden_layer = self.init_hidden(len(batch_text))

#         embeddings = enco(batch_text)

#         # x_packed = pack_padded_sequence(embeddings, batch_first = True)
        
#         # packed_seqs = pack_padded_sequence(embeddings, batch_length)
#         print(embeddings.shape)
#         lstm_output, _ = self.lstm(embeddings, hidden_layer)
#         print(lstm_output.shape)
#         # lstm_output, _ = pad_packed_sequence(lstm_output)

#         # self.relu(lstm_output)
#         # lstm_output, op_lengths = pad_packed_sequence(lstm_output, batch_first = True)

#         lstm_output = self.dropout_layer(lstm_output)
#         print(lstm_output.shape)

#         emissions = self.hidden2tag(lstm_output)
#         # emissions = torch.squeeze(emissions)
#         # emissions = emissions.unsqueeze(0)

#         return emissions
        
#     def loss(self, batch_text, batch_label):
#         # print(len(batch_text))

#         # hidden_layer = self.init_hidden(len(batch_text))

#         # embeddings = enco(batch_text)

#         # # x_packed = pack_padded_sequence(embeddings, batch_first = True)
        
#         # # packed_seqs = pack_padded_sequence(embeddings, batch_length)
#         # lstm_output, _ = self.lstm(embeddings, hidden_layer)
#         # print(lstm_output.shape)
#         # # lstm_output, _ = pad_packed_sequence(lstm_output)

#         # # self.relu(lstm_output)
#         # # lstm_output, op_lengths = pad_packed_sequence(lstm_output, batch_first = True)

#         # lstm_output = self.dropout_layer(lstm_output)
#         # print(lstm_output.shape)

#         emissions = self.generate_emissions(batch_text)
#         batch_label = batch_label.unsqueeze(1)
#         # print(logits.shape)
#         loss = -self.crf_model(emissions, batch_label)

#         return loss
    
#     def predict(self, batch_text):
#         emissions = self.generate_emissions(batch_text)
#         # print(logits.shape)
#         label = self.crf_model.decode(emissions)
#         return label
    
#     def init_hidden(self, batch_size):
#         return (torch.randn(2, 1, self.hidden_dim).to(device), torch.randn(2, 1, self.hidden_dim).to(device))

# %%
import torch
import torch.nn as nn
from TorchCRF import CRF

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_emotions, embedding_dropout=0.2):
        super(EmotionClassifier, self).__init__()
        
        self.audio_dropout = nn.Dropout(embedding_dropout)
        self.video_dropout = nn.Dropout(embedding_dropout)
        self.text_dropout = nn.Dropout(embedding_dropout)

        self.first_linear = nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.relu = nn.ReLU()
        
        self.second_linear_layer = nn.Linear(hidden_size, hidden_size, dtype=torch.float32)
        # Replace Transformer with BiLSTM
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers, 
                              dropout=dropout, bidirectional=True, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, num_emotions)
        self.crf_model = CRF(num_emotions)
        

    def generate_emissions(self, audio_encoding, video_encoding, text_encoding):
        # Concatenate or combine the audio, video, and text encodings
        audio_encoding = audio_encoding.float()
        video_encoding = video_encoding.float()
        text_encoding = text_encoding.float()
        
        print(audio_encoding.shape)
        print(video_encoding.shape)
        print(text_encoding.shape)
        
        audio_encoding = self.audio_dropout(audio_encoding)
        video_encoding = self.video_dropout(video_encoding)
        text_encoding = self.text_dropout(text_encoding)
        
        combined_encoding = torch.cat((audio_encoding, video_encoding, text_encoding), dim=2)
        
        combined_encoding = self.first_linear(combined_encoding)
        combined_encoding = self.relu(combined_encoding)
        combined_encoding = self.second_linear_layer(combined_encoding)
        
        # Pass through BiLSTM
        lstm_output, _ = self.bilstm(combined_encoding)

        # Take the output of the BiLSTM
        emotion_logits = self.linear(lstm_output)
        # Apply a softmax layer
        # emotion_logits = torch.softmax(emotion_logits, dim=2)

        return emotion_logits

    def loss(self, audio_encoding, video_encoding, text_encoding, emotion_labels, padding):

        emissions = self.generate_emissions(audio_encoding, video_encoding, text_encoding)
        emotion_labels = emotion_labels.unsqueeze(1)
        loss = -self.crf_model(emissions, emotion_labels)

        return loss
    
    def predict(self, audio_encoding, video_encoding, text_encoding):
        emissions = self.generate_emissions(audio_encoding, video_encoding, text_encoding)
        label = self.crf_model.decode(emissions)
        return label
    
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim).to('cuda'), torch.randn(2, 1, self.hidden_dim).to('cuda'))

print("BBBBBBBBBBBBBBBBBBBBBBBBB")

# %%
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

# Define your model
# model = EmotionClassifier(input_size=11237, hidden_size=5000, num_layers=2, num_heads=2, dropout=0.2, num_emotions=7)
emotion_model = EmotionClassifier(input_size=768*3, hidden_size=2000, num_layers=3, dropout=0.6, num_emotions=7)
cause_model = EmotionClassifier(input_size=768*3, hidden_size=2000, num_layers=2, dropout=0.6, num_emotions=2)
emotion_model.to("cuda")
cause_model.to("cuda")

weights_tensor = torch.tensor(np.array(indexer.weights)).to("cuda").float()
emotion_criterion = nn.CrossEntropyLoss(
    weight=weights_tensor,
    ignore_index=7
)

cause_criterion = nn.CrossEntropyLoss(ignore_index=-1)

num_epochs = 30
total_steps = len(train_dataloader) * num_epochs

emotion_optimizer = AdamW(emotion_model.parameters(), lr=0.0001)
emotion_lr_scheduler = get_linear_schedule_with_warmup(
    emotion_optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

cause_optimizer = AdamW(cause_model.parameters(), lr=1e-4)
cause_lr_scheduler = get_linear_schedule_with_warmup(
    cause_optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# Define training parameters

# Training loop
min_loss_epoch = -1
min_loss = 1e9
for epoch in (range(num_epochs)):
    emotion_model.train()  # Set the model to training mode
    cause_model.train()
    
    total_loss = 0.0
    total_tokens = 0
    total_correct_emotions = 0
    total_correct_causes = 0
    total_predictions = 0

    for batch in tqdm(train_dataloader):  # Assuming you have a DataLoader for your dataset
        # Extract data from the batch
        audio = batch['audio'].to('cuda')
        video = batch['video'].to('cuda')
        text = batch['text'].to('cuda')
        cause_indices = batch['cause_labels'].to('cuda')
        
        audio_copy = audio.clone().detach()
        video_copy = video.clone().detach()
        text_copy = text.clone().detach()
        
        emotion_indices = batch['emotion_labels'].to('cuda')
        pad_mask = batch['pad_mask'].to('cuda')

        # Forward pass
        # emotion_logits = emotion_model(audio_copy, video_copy, text_copy)

        # Reshape emotion_logits
        # emotion_logits = emotion_logits.view(-1, emotion_logits.size(-1))

        # Flatten emotion_indices (assuming it's a 2D tensor with shape [batch_size, max_sequence_length])
        # emotion_indices = emotion_indices.view(-1)

        # Calculate a mask to exclude padded positions from the loss
        pad_mask = pad_mask.view(-1).float()

        # Calculate the loss, excluding padded positions
        emotion_loss = emotion_model.loss(audio_copy, video_copy, text_copy, emotion_indices, pad_mask)
        # masked_loss = torch.sum(loss * pad_mask) / torch.sum(pad_mask)
        # masked_loss = emotion_loss# *pad_mask
        # Backpropagation and optimization
        
        
        # cause_logits = cause_model(audio, video, text)
        # cause_logits = cause_logits.view(-1, cause_logits.size(-1))
        # cause_indices = cause_indices.view(-1)
        
        cause_loss = cause_model.loss(audio, video, text, cause_indices, pad_mask)
        masked_loss += cause_loss
        
        emotion_optimizer.zero_grad()
        cause_optimizer.zero_grad()
        
        masked_loss.backward()
        
        emotion_optimizer.step()
        cause_optimizer.step()        

        total_loss += masked_loss.item()
        total_tokens += torch.sum(pad_mask).item()
        
        predicted_emotions = emotion_model.predict(audio, video, text)
        correct_predictions_emotions = ((predicted_emotions == emotion_indices) * pad_mask).sum().item()

        predicted_causes = cause_model.predict(audio, video, text)
        correct_predictions_causes = ((predicted_causes == cause_indices) * pad_mask).sum().item()
        
        total_correct_emotions += correct_predictions_emotions
        total_correct_causes += correct_predictions_causes
        total_predictions += torch.sum(pad_mask).item()  # Batch size
        
    
    emotion_lr_scheduler.step()
    cause_lr_scheduler.step()
    
    emotion_model.eval()  # Set the model to evaluation mode
    cause_model.eval()
    
    total_val_loss = 0.0
    total_val_tokens = 0
    total_val_correct_emotions = 0
    total_val_correct_causes = 0
    total_val_predictions = 0
    true_labels_emotion = []
    predicted_labels_emotion = []
    true_labels_cause = []
    predicted_labels_cause = []
    padded_labels = []

    with torch.no_grad():
        for val_batch in tqdm(validation_dataloader):
            audio = val_batch['audio'].to('cuda')
            video = val_batch['video'].to('cuda')
            text = val_batch['text'].to('cuda')
            emotion_indices = val_batch['emotion_labels'].to('cuda')
            cause_indices = val_batch['cause_labels'].to('cuda')
            pad_mask = val_batch['pad_mask'].to('cuda')
            
            audio_copy = audio.clone().detach()
            video_copy = video.clone().detach()
            text_copy = text.clone().detach()

            # emotion_logits = emotion_model(audio_copy, video_copy, text_copy)

            # Reshape emotion_logits
            # emotion_logits = emotion_logits.view(-1, emotion_logits.size(-1))

            # Flatten emotion_indices (assuming it's a 2D tensor with shape [batch_size, max_sequence_length])
            # emotion_indices = emotion_indices.view(-1)

            pad_mask = pad_mask.view(-1)   

            # Calculate the loss, excluding padded positions
            val_loss = emotion_model.loss(audio_copy, video_copy, text_copy, emotion_indices, pad_mask)
            masked_loss = val_loss #torch.sum(val_loss * pad_mask) / torch.sum(pad_mask)
            
            # cause_logits = cause_model(audio, video, text)
            # cause_logits = cause_logits.view(-1, cause_logits.size(-1))
            # cause_indices = cause_indices.view(-1)
            cause_loss = cause_criterion(audio, video, text, cause_indices, pad_mask)
            masked_loss += cause_loss
            
            total_val_loss += masked_loss.item()
            total_val_tokens += torch.sum(pad_mask).item()
            
            predicted_emotions_val = emotion_model.predict(audio, video, text)
            correct_predictions_val = ((predicted_emotions_val == emotion_indices) * pad_mask).sum().item()
            total_val_correct_emotions += correct_predictions_val
            
            predicted_causes_val = cause_model.predict(audio, video, text)
            correct_predictions_causes_val = ((predicted_causes_val == cause_indices) * pad_mask).sum().item()
            total_val_correct_causes += correct_predictions_causes_val
            
            total_val_predictions += torch.sum(pad_mask).item()

            # Store true and predicted labels for F1 score calculation
            true_labels_emotion.extend(emotion_indices.cpu().numpy())
            predicted_labels_emotion.extend(predicted_emotions_val.cpu().numpy())
            
            true_labels_cause.extend(cause_indices.cpu().numpy())
            predicted_labels_cause.extend(predicted_causes_val.cpu().numpy())
            padded_labels.extend(pad_mask.cpu().numpy())

    final_true_labels_emotion = [label for label, pad in zip(true_labels_emotion, padded_labels) if pad == 1]
    final_predicted_labels_emotion = [label for label, pad in zip(predicted_labels_emotion, padded_labels) if pad == 1]
    
    final_true_labels_cause = [label for label, pad in zip(true_labels_cause, padded_labels) if pad == 1]
    final_predicted_labels_cause = [label for label, pad in zip(predicted_labels_cause, padded_labels) if pad == 1]
    
    emotion_classification_rep = classification_report(final_true_labels_emotion, final_predicted_labels_emotion)
    cause_classification_rep = classification_report(final_true_labels_cause, final_predicted_labels_cause)
    
    # Calculate and print the average loss for this epoch
    avg_loss = total_loss / total_tokens
    avg_val_loss = total_val_loss / total_val_tokens
    
    print("===============================")
    print("Training data metrics")
    print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {avg_loss}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] Accuracy: {total_correct_emotions / total_predictions}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] Accuracy: {total_correct_causes / total_predictions}")
    
    print("VALIDATION METRICS")
    print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss}")
    print(emotion_classification_rep)
    print(cause_classification_rep)
    print("===============================")

    torch.save(emotion_model.state_dict(), f"/tmp/semeval24_task3/final_models/emotion_models/emotion_model_{epoch:02}.pt")
    torch.save(cause_model.state_dict(), f"/tmp/semeval24_task3/final_models/cause_models/cause_model_{epoch:02}.pt")
    if avg_val_loss < min_loss:
        min_loss = avg_val_loss
        min_loss_epoch = epoch

print("Training complete!")
print("CCCCCCCCCCCCCCCCCCCC")
print(min_loss_epoch)
