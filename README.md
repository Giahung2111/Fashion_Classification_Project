# Fashion Classification Project

This project aims to classify images of fashion items using a "MLP Linear Model", "MLP Non-Linear Model", "Convolutional neural network model (Tiny VGG)" built with PyTorch.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd Fashion_Classification_Project
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

The configuration settings for the project are located in `configs/config_settings.py`. You can modify the parameters such as batch size, learning rate, number of epochs, etc.

## Training

To train the model, run the following command:
```sh
python [train.py](http://_vscodecontentref_/13)
```
This will train the model and save the trained model to checkpoints/model.pth.

## Evaluation
To evaluate the model, run the following command:
```sh
python [evaluate.py](http://_vscodecontentref_/14)
```
This will load the trained model and print the evaluation results.

## Inference
To perform inference using the trained model, run the following command:
```sh
python [infer.py](http://_vscodecontentref_/15)
```

## Jupyter Notebook
You can also explore the project and run experiments using the provided Jupyter notebook test.ipynb.

## Logging
Training and evaluation logs are saved in the logs/ directory:

logs/training_logs.txt: Contains logs for the training process.
logs/evaluatue_logs.txt: Contains logs for the evaluation process.

## Project Files
train.py: Script to train the model.
evaluate.py: Script to evaluate the model.
infer.py: Script to perform inference using the trained model.
configs/config_settings.py: Configuration settings for the project.
models/model.py: Contains the model definition.
utils/data_utils.py: Utility functions for data processing.
utils/evaluate_utils.py: Utility functions for model evaluation.
utils/train_utils.py: Utility functions for model training.
datasets/custom_dataset.py: Custom dataset definitions.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
PyTorch
Torchvision
Torchmetrics