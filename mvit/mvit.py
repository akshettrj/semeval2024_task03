import argparse
from torchvision.io.video import read_video
import torchvision.models.video as yo 
# import mvit_v2_s, MViT_V2_S_Weights


# take the video path as an argument
parser = argparse.ArgumentParser(description='Process some videos.')
parser.add_argument('--directory', type=str, help='Directory containing videos')
args = parser.parse_args()

vid, _, _ = read_video(args.directory)
vid = vid[:32]  # optionally shorten duration

# Step 1: Initialize model with the best available weights
weights = yo.MViT_V2_S_Weights.KINETICS400_V1
model = yo.mvit_v2_s(weights=weights).eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(vid).unsqueeze(0)

# get the video features
features = model(batch).squeeze(0)

print("features.shape : ",features.shape)
print("features : ",features)


# # Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# label = prediction.argmax().item()
# score = prediction[label].item()
# category_name = weights.meta["categories"][label]
# print(f"{category_name}: {100 * score}%")
