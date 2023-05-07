# DCPGRN
Dynamic Climate Pattern Graph Recurrent Network (DCPGRN) for meteorological forecasting
This is a official pyTorch implementation of DCPGRN

![image](https://user-images.githubusercontent.com/44642002/236671180-822846e7-b3fd-4dce-b01c-fa383b6d5338.png)
![image](https://user-images.githubusercontent.com/44642002/236467448-15e556f8-d9b8-4407-8bb0-8c5373b827eb.png)



## Basic Requirements
torch>=1.7.0
torch-geometric-temporal

Dependency can be installed using the following command:

pip install -r requirements.txt

## Data Preparaion
The four datasets after preprocessed are available at [Google Drive]().

Download the dataset and copy it into data/ dir. And Unzip them.

To execute the baseline experiment, you can change the "graph_model", 


## Training the Model
The configuration is set in config.yaml file for training process. Run the following commands to train the target model.

python train.py

We are further developing the distributed version for a larger scale GNN model.

## Acknowledgement
The project is developed based on [PM2.5-GNN](https://github.com/shuowang-ai/PM2.5-GNN) and [IDGL](https://github.com/hugochan/IDGL) for dynamic graph.
