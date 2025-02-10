# ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge
This repository contains the code and description of our solution as well as the report and the final presentation slide.

## Requirements
Our implementation is based on the tensorflow version of [RouteNet model](https://github.com/BNN-UPC/GNNetworkingChallenge/tree/2021_Routenet_TF). The main librairies
used are:

• numpy==1.19.5

• pandas==1.3.2

• tensorflow==2.4.2

• networkx==2.5.1

All of them can be installed with ```pip install```. The python version is Python 3.8.8.

## Proposed Solution
Our solution is made of the combination of two independent GNN models. For the training, both models is trained separately (sequentially or in parallel). Here is its architecture.

<img id="fig1" src="./assets/SOFGNN_architecture.PNG" width="100%" height="100%" />

##  Training and Evaluation
### 1 - Datasets

Download the [Traning, Validation and Test Datasets](https://bnn.upc.edu/challenge/gnnet2021/dataset/)  made available by the [Barcelona Neural Networking Center](https://bnn.upc.edu/), then save them in a folder named Dataset following the specifications contained in the [config.ini](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/config.ini) file. Update the [[DIRECTORIES]](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/config.ini#L1)  paths  to match with your envionment.

### 2 - Train the model

• To train model 1: execute the [main_1.py](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/main_1.py) script with ```python main_1.py```. This will train model 1 and save the model after each epoch in a folder named trained model1.

• To train model 2: execute the [main_2.py](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/main_2.py)  script with ```python main_2.py```. This will train model 2 and save the model after each epoch in a folder named trained model2.

### 3 - Select the Best model

• For model 1: run the [find top models 1.py](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/find_top_models_1.py) script with ```python find_top_models_1.py```.

• For model 2: run the [find top models 2.py](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/find_top_models_2.py) script with ```python find_top_models_2.py```.

### 4 - Evaluate the model

To evaluate the ensemble model, run the [evaluate.py](https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-001-SOFGNN-Graph-Neural-Networking-Challenge/blob/main/code/evaluate.py) script with  ```python evaluate.py```.

# Credits
This project is the result of the collaboration between:

[Brigitte Jaumard](https://users.encs.concordia.ca/~bjaumard/) - Computer Science and Software Engineering Dpt, Concordia University, Montreal (Qc) Canada

Charles Boudreau - Computer Science and Software Engineering Dpt, Concordia University, Montreal (Qc) Canada

[Junior Momo Ziazet](https://ca.linkedin.com/in/junior-momo-ziazet-710755140) - Computer Science and Software Engineering Dpt, Concordia University, Montreal (Qc) Canada

[Huy Duong](https://ca.linkedin.com/in/huy-q-duong) - Centre de Recherche Informatique de Montréal (CRIM), Montreal (Qc) Canada
