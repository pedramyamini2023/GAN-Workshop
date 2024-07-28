import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from generator import Generator  # Import the Generator class from generator.py

def load_model(model_path, device):
    generator = Generator()
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['state_dict'])
    generator.eval()
    return generator

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the same size used during training
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    return input_tensor

def generate_image(generator, input_tensor, device):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output_tensor = generator(input_tensor).cpu()
    return output_tensor

def save_image(output_tensor, output_path):
    output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
    output_tensor = (output_tensor * 0.5) + 0.5  # Denormalize to [0, 1]
    output_image = transforms.ToPILImage()(output_tensor)
    output_image.save(output_path)
    return output_image

def main():
    parser = argparse.ArgumentParser(description='Generate an output image from an input image using a pre-trained Pix2Pix model.')
    parser.add_argument('input_image_path', type=str, help='Path to the input image')
    parser.add_argument('output_image_path', type=str, help='Path to save the output image')
    parser.add_argument('--model_path', type=str, default='gen.pth.tar', help='Path to the pre-trained model (default: gen.pth.tar)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (default: cpu)')

    args = parser.parse_args()

    generator = load_model(args.model_path, args.device)
    input_tensor = preprocess_image(args.input_image_path)
    output_tensor = generate_image(generator, input_tensor, args.device)
    output_image = save_image(output_tensor, args.output_image_path)

    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
