from PIL import Image
import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from flask import Flask, request, jsonify, send_file, redirect
import time
from flask_cors import CORS
import io
from torchvision.transforms import transforms

predicted_name = None

app = Flask(__name__)
CORS(app, supports_credentials=True, methods=["GET", "POST", "PUT", "DELETE"])

classes = {'Abbott’s Babbler Malacocincla abbotti': 0, 'Black Bittern (Dupetor flavicollis)': 1, 'Blue-eared Kingfisher Alcedo meninting': 2, 'Blue-naped Pitta Pitta nipalensis': 3, 'Broad-billed Warbler Tickellia hodgsoni': 4, 'Cheer Pheasant (Catreus wallichii)': 5, 'Chestnut Munia Lonchura atricapilla': 6, 'Cinereous Vulture Aegypius monachus': 7, 'Golden Babbler Stachyris chrysaea': 8, 'Gould’s Shortwing Brachypteryx stellata': 9, 'Great Bittern Botaurus stellaris': 10, 'Great Hornbill (Buceros bicornis)': 11, 'Great Slaty Woodpecker Mulleripicus pulverulentus': 12, 'Ibisbill Ibidorhyncha struthersii': 13, 'Indian Courser Cursorius coromandelicus': 14, 'Indian Grassbird - Graminicola bengalensis': 15, 'Indian Nightjar Caprimulgus asiaticus': 16, 'Knob-billed Duck Sarkidiornis melanotos': 17, 'Northern Pintail Anas acuta': 18, 'Painted Stork Mycteria leucocephala': 19,
           'Purple Cochoa Cochoa purpurea': 20, 'Red-headed Trogon Harpactes erythrocephalus': 21, 'Red-headed Vulture Sarcogyps calvus': 22, 'Red-necked Falcon Falco chicquera': 23, 'Ruby-cheeked Sunbird Anthreptes singalensis': 24, 'Rusty-fronted Barwing Actinodura egertoni': 25, 'Saker Falcon Falco cherrug': 26, 'Silver-eared Mesia Leiothrix argentauris': 27, 'Slaty-legged Crake Rallina eurizonoides': 28, 'Spot-bellied Eagle Owl Bubo nipalensis': 29, 'Sultan Tit Melanochlora sultanea': 30, 'Swamp Francolin Francolinus gularis': 31, 'Tawny-bellied Babbler Dumetia hyperythra': 32, 'Thick-billed Green Pigeon Treron curvirostra': 33, 'White-throated Bulbul Alophoixus flaveolus': 34, 'White-throated Bushchat Saxicola insignis': 35, 'Yellow-rumped Honeyguide - Indicator xanthonotus': 36, 'Yellow-vented Warbler Phylloscopus cantator': 37}
classes_names = list(classes.keys())
class_labels = list(classes.values())

prediction_mapping = [
    'Abbott’s Babbler Malacocincla abbotti = मोटोठूँडे भ्याकुर',
    'Black Bittern (Dupetor flavicollis) = कालो जूनबकुल्ला',
    'Blue-eared Kingfisher Alcedo meninting = सानो माछरी प्याकुली',
    'Blue-naped Pitta Pitta nipalensis = नीलकन्ठ पिट्टा',
    'Broad-billed Warbler Tickellia hodgsoni = चरीमा पाती मौनी',
    'Cheer Pheasant (Catreus wallichii) = काली मोनाल',
    'Chestnut Munia Lonchura atricapilla = कोटेरो मुनियाँ',
    'Cinereous Vulture Aegypius monachus = राज गिद्ध',
    'Golden Babbler Stachyris chrysaea = सुन्दर पुर्णक',
    'Gould’s Shortwing Brachypteryx stellata = थोप्ले लघुपंख',
    'Great Bittern Botaurus stellaris = कालो जूनबकुल्ला',
    'Great Hornbill (Buceros bicornis) = राजधनेश',
    'Great Slaty Woodpecker Mulleripicus pulverulentus = बडी सानो कठठारी',
    'Ibisbill Ibidorhyncha struthersii = तिलहरी चरा',
    'Indian Courser Cursorius coromandelicus = गाजले धावक',
    'Indian Grassbird (Graminicola bengalensis) = झारे खेती मौनी',
    'Indian Nightjar Caprimulgus asiaticus = चुकचुके चैतेचरा',
    'Knob-billed Duck Sarkidiornis melanotos = नकटा',
    'Northern Pintail Anas acuta = सुइरोपुछ्रे',
    'Painted Stork Mycteria leucocephala = लालटाउके गरुड',
    'Purple Cochoa Cochoa purpurea = बैजनी कचोवा',
    'Red-headed Trogon Harpactes erythrocephalus = रक्त्त शिर',
    'Red-headed Vulture Sarcogyps calvus = सुन गिद्ध',
    'Red-necked Falcon Falco chicquera = रातोटाउके बौँडाइ',
    'Ruby-cheeked Sunbird Anthreptes singalensis = प्याजीकाने बुङ्गेचरा',
    'Rusty-fronted Barwing Actinodura egertoni = कैलोतालु वनचाहर',
    'Saker Falcon Falco cherrug = तोप बाज',
    'Silver-eared Mesia Leiothrix argentauris = चाँदीकाने मिसीया',
    'Slaty-legged Crake Rallina eurizonoides = देउकौवा',
    'Spot-bellied Eagle Owl Bubo nipalensis = महाकौशिक',
    'Sultan Tit Melanochlora sultanea = स्वर्णचूल राजचिचिल्कोटे',
    'Swamp Francolin Francolinus gularis = सिमतित्रा',
    'Tawny-bellied Babbler Dumetia hyperythra = कैलो घाँसेभ्याकुर',
    'Thick-billed Green Pigeon Treron curvirostra = मोटोठूँडे हलेसो',
    'White-throated Bulbul Alophoixus flaveolus =  सेतोकण्ठे जुरेली',
    'White-throated Bushchat Saxicola insignis = सेतोकण्ठे धिप्सी',
    'Yellow-rumped Honeyguide - Indicator xanthonotus = पीतनिर्गम फिस्टो',
    'Yellow-vented Warbler Phylloscopus cantator = मोटोठूँडे भ्याकुर'
]

classes_names = list(classes.keys())
class_labels = list(classes.values())

# Model Definition


class ConvNet(nn.Module):
    def __init__(self, num_classes=38, kernel_size_conv1=3, in_channels=3,
                 out_channels_conv1=12, out_channels_conv2=20, kernel_size_conv2=3,
                 out_channels_conv3=32, kernel_size_conv3=3, num_features_batchnorm=32,
                 kernel_size_maxpool=2, dropout_prob=0.3):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_conv1,
                               kernel_size=kernel_size_conv1, stride=1, padding=1)
        self.btch1 = nn.BatchNorm2d(num_features=out_channels_conv1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_maxpool)

        self.conv2 = nn.Conv2d(in_channels=out_channels_conv1, out_channels=out_channels_conv2,
                               kernel_size=kernel_size_conv2, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.conv3 = nn.Conv2d(in_channels=out_channels_conv2, out_channels=out_channels_conv3,
                               kernel_size=kernel_size_conv3, stride=1, padding=1)
        self.btch3 = nn.BatchNorm2d(num_features=out_channels_conv3)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)  # Regularization drop_out

        # Fully connected layer
        self.fc = nn.Linear(in_features=75 * 75 *
                            out_channels_conv3, out_features=num_classes)

    def forward(self, input):
        # Convolutional layers
        ot_put = self.conv1(input)
        ot_put = self.btch1(ot_put)
        ot_put = self.relu1(ot_put)
        ot_put = self.pool(ot_put)

        ot_put = self.conv2(ot_put)
        ot_put = self.relu2(ot_put)
        ot_put = self.dropout1(ot_put)

        ot_put = self.conv3(ot_put)
        ot_put = self.btch3(ot_put)
        ot_put = self.relu3(ot_put)
        ot_put = self.dropout2(ot_put)

        # Reshape and fully connected layer
        ot_put = ot_put.view(-1, 75 * 75 * 32)
        ot_put = self.fc(ot_put)

        return ot_put


def prediction(img_path, trnsfrm, model, class_labels, device):
    image = Image.open(img_path).convert('RGB')
    image_tensor = trnsfrm(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    image_tensor = image_tensor.to(device)  # Move tensor to device

    output = model(image_tensor)
    _, index = output.data.cpu().max(1)  # Get the index of the max log-probability
    label_index = index.item()

    # Convert label index to the corresponding label
    pred_label = list(class_labels)[list(
        class_labels).index(label_index)]
    return pred_label


# Define the device for model inference
device = torch.device('cpu')  # Change to 'cuda' if you have GPU support

# Load the pre-trained model
model = ConvNet()
model.load_state_dict(torch.load(
    "best_cHpnt10fld500.pth", map_location=device))
model.eval()

# Transform for preprocessing the input image
trnsfrm = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Create the 'uploads' folder if it doesn't exist
uploads_folder = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)


@app.route("/predict", methods=["POST"])
def prediction_route():
    global predicted_name

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file found'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image to the 'uploads' folder with the name "upload.png"
        filename = "upload.png"
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        print("Image saved to:", filepath)  # Debugging

        # Add your image processing or prediction logic here
        predicted_label = prediction(
            filepath, trnsfrm, model, class_labels, device)

        prediction_name = prediction_mapping[predicted_label]

        # Set the predicted_name variable
        predicted_name = prediction_name

        return jsonify({"predicted_name": predicted_name})

    except Exception as e:
        print(f'Error: {str(e)}')  # Add this line for additional information
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/get_text')
def get_text():
    global predicted_name

    # Return the predicted name if available
    if predicted_name:
        response = {"predicted_name": predicted_name}
        predicted_name = None  # Reset the variable
        return jsonify(response)

    return jsonify({"predicted_name": None})


@app.route('/delete-image', methods=['DELETE'])
def delete_image():
    global predicted_name
    # print(predicted_name)
    directory = "E:/final/uploads"
    print(directory)
    try:
        file_names = os.listdir(directory)
    except FileNotFoundError:
        return jsonify({'error': 'Directory not found'}), 404

    if not file_names:
        return jsonify({'error': 'No files to delete'}), 404

    file_to_delete = os.path.join(directory, file_names[0])

    try:
        # Set predicted_name to "none"
        predicted_name = "Enter Image"
        os.remove(file_to_delete)

        return jsonify({'message': 'Image deleted successfully'}), 204
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='localhost', port=8000)
