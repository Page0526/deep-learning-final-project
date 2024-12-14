import onnxruntime
import numpy as np
from torchvision import transforms
from PIL import Image
import argparse

class ImageClassifierONNX:
    def __init__(self, model_path, input_size=(1, 64, 64), num_classes=4):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_size = input_size
        self.num_classes = num_classes

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((input_size[1], input_size[2]))
        ])

    def preprocess(self, image):
    
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        transformed_image = self.data_transform(image)
        input_data = transformed_image.unsqueeze(0).numpy()

        return input_data

    def predict(self, image):
        input_data = self.preprocess(image)

        inputs = {self.session.get_inputs()[0].name: input_data}

        outputs = self.session.run(None, inputs)[0]
        predicted_class = np.argmax(outputs, axis=1)[0]
        confidence = np.max(outputs, axis=1)[0]

        return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer a brain tumor classification model")
    parser.add_argument("-model_name", type=str, required=True, help="Name of the model to infer (resnet, vgg16, densenet, capsnet)")
    parser.add_argument("-image_path", type=str, required=True, help="Path to the image to infer")


    args = parser.parse_args()

    model_path = "/model_cpt/" + args.model_name +  ".onnx"  
    classifier = ImageClassifierONNX(model_path)


    image = Image.open(args.image_path).convert("RGB")
    predicted_class, confidence = classifier.predict(image)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")