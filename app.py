from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
from torchvision import datasets
import timm
import requests
from bs4 import BeautifulSoup
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to save uploaded images

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define your model architecture
model = timm.create_model('mobilenetv4_conv_medium.e500_r256_in1k', num_classes=50, pretrained=False)
model.load_state_dict(torch.load("FINAL_fulll_2.pth", map_location=device))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset to get class labels
data_dir = "C:/Users/Dell/Desktop/ML_CP/DATASET/IHDS_dataset"
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=preprocess)
class_labels = train_dataset.classes  # Extract class labels

@app.route('/')
def home():
    return render_template("index.html")  # Ensure index.html exists in the templates folder

def get_heritage_info(site_name):
    """Fetch a brief description of the heritage site from the web."""
    try:
        search_url = f"https://en.wikipedia.org/wiki/{site_name.replace(' ', '_')}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraph = soup.find('p')  # Get the first paragraph
        if paragraph:
            return paragraph.get_text(strip=True)
        else:
            return "No information found for this site."
    except Exception as e:
        return f"Error fetching information: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']

    try:
        # Save the uploaded image to the upload folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Open and preprocess the image
        image = Image.open(file.stream).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)  # Send image to the same device as the model

        # Predict
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Compute class probabilities
            _, predicted = torch.max(output, 1)
            predicted_class = class_labels[predicted.item()]

        # Sort class-wise probabilities in descending order
        sorted_probabilities = sorted(
            [(class_labels[i], float(probabilities[i])) for i in range(len(class_labels))],
            key=lambda x: x[1],
            reverse=True
        )

        # Print sorted probabilities to the terminal
        print("\nClass-Wise Probabilities (Descending Order):")
        for label, prob in sorted_probabilities:
            print(f"{label}: {prob:.4f}")

        # Get related heritage site information
        heritage_info = get_heritage_info(predicted_class)

        return render_template(
            "result.html",
            uploaded_image=image_path,
            predicted_class=predicted_class,
            heritage_info=heritage_info
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
