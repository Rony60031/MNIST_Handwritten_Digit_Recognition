# MNIST Handwritten Digit Recognition

A Convolutional Neural Network (CNN) project built with Python and PyTorch to recognize handwritten digits from 0 to 9 using the MNIST dataset.

## Technologies
- Python 3.13
- PyTorch
- Torchvision
- NumPy
- Matplotlib

## Dataset
- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 x 28
- Classes: 0-9

## Model Architecture
- Two Convolutional Layers
- ReLU Activation
- Max Pooling
- Two Fully Connected Layers
- Adam Optimizer
- Cross Entropy Loss

## Training Configuration
- Epochs: 20
- Batch Size: 512
- Device: CPU

## Results
- Final Training Accuracy: 99.91%
- Final Test Accuracy: 98.91%
- Best Test Accuracy: 99.07%
- Final Training Loss: 0.0031
- Training Time: 605.2 seconds

## Project Structure
- mnist_handwritten_experiment.py: Main experiment code
- MNIST_Experiment_Results/: Model, graphs and training results
- screenshots/: Experiment screenshots
- report/: Experiment report

## Run the Project
Install dependencies:

pip install torch==2.6.0 torchvision==0.21.0
pip install matplotlib numpy

Run:

python mnist_handwritten_experiment.py

Note: Place the MNIST dataset in the required local data directory before running the script.

## Author
Rony60031

GitHub: https://github.com/Rony60031
