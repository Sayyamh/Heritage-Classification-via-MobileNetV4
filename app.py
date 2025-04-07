from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
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

# Hardcoded class labels (make sure order matches training)
class_labels = [
    'ibrahim roza', 'AMRUTESHWARA TEMPLE, ANNIGERI', 'Brahmeshwar temple,kikkeri',
    'KAMAL BASTI, BELAGAVI', 'Kiatabeshwar temple, Kubatur', 'Kumaraswamy temple,Sandur,Hospet',
    'Mahadev Temple Tambdisurla Goa', 'Someshwar temple kaginele', 'Trikuteshwara temple, Gadag',
    'chennakeshwara temple. belur', 'Ambigera gudi complex,Aihole', 'Billeshwar Temple Hanagal',
    'Channakeshwa_temple_aralguppe', 'Digambar Basti, Belgum', 'Doddabasappa temple,Gadag',
    'GALAGANATH TEMPLE HAVERI', 'Goudaragudi temple, aihole', 'HAZARARAMA_TEMPLE_HAMPI',
    'HOYSALESHWAR TEMPLE, HALEBEEDU', 'Jain_Basadi_Bilagi', 'KAADASIDHESHWAR TEMPLE , PATTADAKAL',
    'KAPPECHENIKESHWARA_TEMPLE_HASSAN', 'KEDARESHWARA_TEMPLE_HASSAN', 'KOTILINGESHWARA, KOTIPUR, HANAGAL',
    'Keshava Temple Somanathapur,Mysore', 'Koravangala Temple,Hassan', 'Kunti Temple Complex, Aihole',
    'LAKSHMIKANT TEMPLE,NANJANGUDU,MYSORE', 'LOTUS MAHAL, HAMPI', 'Lady_of_mount_Goa',
    'MADHUKESHWARA TEMPLE, BANAVASI', 'MOOLE SHANKARESHWARA TEMPLE,TURUVEKERE', 'Mallikarjuna Temple,Mandya',
    'NAGARESHWARA_TEMPLE,BANKAPUR', 'PAPANATH TEMPLE_PATTADAKAL', 'Rameshwar_temple',
    'SHIVA BASADI(SHRAVANBELAGOLA)', 'Sangameshwar_Pattadakal', 'Someshwara Temple,Lakshmeshwara',
    'TARAKESHWRARA_TEMPLE_HANGAL', 'TWIN TOWER TEMPLE SUDI', 'Veerabhadreshwara temple,Hangal',
    'agra_fort', 'aihole', 'hampi monolithic bull', 'hampi_chariot',
    'kadambeshwara temple Rattihalli,Haveri', 'mahabodhi_temple', 'mahadeva temple ,ittagi', 'safa masjid _belgaum'
]

@app.route('/')
def home():
    return render_template("index.html")  # Ensure index.html exists in the templates folder

def get_heritage_info(site_name):
    """Fetch a brief description of the heritage site from Wikipedia."""
    try:
        search_url = f"https://en.wikipedia.org/wiki/{site_name.replace(' ', '_')}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraph = soup.find('p')
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
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Preprocess image
        image = Image.open(file.stream).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted = torch.max(output, 1)
            predicted_class = class_labels[predicted.item()]

        # Sorted class-wise probabilities
        sorted_probabilities = sorted(
            [(class_labels[i], float(probabilities[i])) for i in range(len(class_labels))],
            key=lambda x: x[1],
            reverse=True
        )

        print("\nClass-Wise Probabilities (Descending Order):")
        for label, prob in sorted_probabilities:
            print(f"{label}: {prob:.4f}")

        # Get heritage info
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
