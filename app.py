import numpy as np
import onnxruntime as rt
from PIL import Image
from flask import Flask, render_template, request
from flask import jsonify
from torchvision.transforms import transforms

app = Flask(__name__)

# Load the ONNX model
session = rt.InferenceSession("models/model.onnx")


def crop_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = (width + height) / 2
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = (height + width) / 2
    image = image.crop((left, top, right, bottom))
    return image


# Define function to preprocess the image
def preprocess_image(image):
    image = crop_image(image)
    image = image.resize((256, 256))  # Resize the image to match the model input size
    image = transforms.ToTensor()(image)
    image = image.float()
    image = np.array(image)
    return image.reshape(1, 3, 256, 256)


# Define function to make predictions
def predict(image):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = preprocess_image(image)
    result = session.run([output_name], {input_name: input_data})
    return result[0][0]


# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    image = Image.open(file.stream).convert("RGB")
    predicted_value = predict(image)

    # Here you can do whatever you want with the prediction result, e.g., return it as JSON
    if predicted_value > 0:
        prediction = "Synkinesis detected"
    else:
        prediction = "Synkinesis not detected"

    # Return prediction result as JSON
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
