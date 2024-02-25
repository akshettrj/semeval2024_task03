import os

DATA_DIR = "/tmp/semeval24_task3"

AUDIO_EMBEDDINGS_FILEPATH = "/tmp/semeval24_task3/audio_embeddings/audio_embeddings_microsoft_wavlm-base-plus-sd.pkl"
VIDEO_EMBEDDINGS_FILEPATH = "/tmp/semeval24_task3/video_embeddings/final_embeddings.pkl"
TEXT_EMBEDDINGS_FILEPATH = os.path.join(DATA_DIR, "text_embeddings", "text_embeddings_roberta_base_emotion.pkl")
