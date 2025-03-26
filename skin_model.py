import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

model_name = "Tuu-invitrace/skin_decease"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

def predict_skin_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(image)  
        probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)  
        predicted_class_idx = probabilities.argmax().item()  
        confidence = probabilities[predicted_class_idx].item() * 100  

    labels = model.config.id2label
    predicted_disease = labels.get(predicted_class_idx, "Unknown Disease")

    return predicted_disease, round(confidence, 2)
