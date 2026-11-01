import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD CLASSES --------
with open("classes.json", "r") as f:
    class_names = json.load(f)

# -------- MODEL --------
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

model.load_state_dict(torch.load("alz_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------- HOOK --------
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -------- INPUT --------
image_path = input("Enter image path: ")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# -------- PREDICTION --------
output = model(input_tensor)
pred = torch.argmax(output).item()

# -------- BACKPROP --------
model.zero_grad()
output[0, pred].backward()

# -------- GRAD-CAM --------
grads = gradients[0].detach().cpu().numpy()[0]
fmap = features[0].detach().cpu().numpy()[0]

weights = np.mean(grads, axis=(1,2))

cam = np.zeros(fmap.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * fmap[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224,224))

# Normalize
cam = cam - np.min(cam)
cam = cam / (np.max(cam) + 1e-8)

# -------- HEATMAP --------
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
img_np = np.array(img.resize((224,224)))
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

# -------- SAVE --------
cv2.imwrite("gradcam_heatmap.jpg", heatmap)
cv2.imwrite("gradcam_overlay.jpg", overlay)

# -------- CONFIDENCE --------
prob = torch.nn.functional.softmax(output, dim=1)
confidence = prob[0][pred].item() * 100

# -------- REGION ANALYSIS --------
h, w = cam.shape

center = cam[h//4:3*h//4, w//4:3*w//4]
top = cam[:h//4, :]
bottom = cam[3*h//4:, :]
left = cam[:, :w//4]
right = cam[:, 3*w//4:]

regions = {
    "center (hippocampus area)": center.mean(),
    "top (frontal lobe)": top.mean(),
    "bottom (temporal region)": bottom.mean(),
    "left hemisphere": left.mean(),
    "right hemisphere": right.mean()
}

important_region = max(regions, key=regions.get)
activation_strength = regions[important_region]

# -------- ACTIVATION % --------
active_pixels = np.sum(cam > 0.6)
total_pixels = cam.size
activation_percent = (active_pixels / total_pixels) * 100

# -------- OUTPUT --------
print("\n========== RESULT ==========")
print("Prediction:", class_names[pred])
print(f"Confidence: {confidence:.2f}%")

print("\n--- Grad-CAM Outputs ---")
print("Saved: gradcam_heatmap.jpg")
print("Saved: gradcam_overlay.jpg")

print("\n--- Explainable AI Insights ---")
print("Most Activated Region:", important_region)
print(f"Activation Strength: {activation_strength:.4f}")
print(f"Activation Spread: {activation_percent:.2f}% of brain")

# -------- DYNAMIC EXPLANATION --------
print("\nExplanation:")

# Base explanation from prediction (medical meaning)
if class_names[pred] == "NonDemented":
    print("Brain structure appears normal with no signs of degeneration.")

elif class_names[pred] == "VeryMildDemented":
    print("Very early signs of structural change detected in memory-related regions.")

elif class_names[pred] == "MildDemented":
    print("Noticeable structural shrinkage detected in brain regions associated with memory.")

else:
    print("Significant degeneration observed in key brain regions.")

# Add explainability insight (from Grad-CAM)
print(f"\nModel focused on {important_region}.")

if activation_percent < 15:
    print("Activation is localized → model is confident about a specific region.")
elif activation_percent < 30:
    print("Activation is moderately spread → multiple regions contribute to decision.")
else:
    print("Activation is widely spread → broader brain involvement detected.")
# -------- DISPLAY --------
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_np)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()