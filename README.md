# DCPGRN
Dynamic Climate Pattern Graph Recurrent Network (DCPGRN) for meteorological forecasting
This is a official pyTorch implementation of DCPGRN

![image](https://github.com/xzwbsz/DCPGRNcode/assets/44642002/8a2dc354-1abb-4a2d-8660-b8cea6cbbbca)


![image](https://user-images.githubusercontent.com/44642002/236467448-15e556f8-d9b8-4407-8bb0-8c5373b827eb.png)



## Basic Requirements
torch>=1.7.0
torch-geometric-temporal

Dependency can be installed using the following command:

pip install -r requirements.txt

## Data Preparaion
The four datasets after preprocessed are available at [Google Drive](https://drive.google.com/drive/folders/18e9plTz8BmWdnw6IExFZyn0_0VY2E5-X?usp=sharing).

Download the dataset and copy it into data/ dir. And Unzip them.

To execute the baseline experiment, you can change the "graph_model", 


## Training the Model
The configuration is set in config.yaml file for training process. Run the following commands to train the target model.

python train.py

We are further developing the distributed version for a larger scale GNN model.

## Experiment Results
We compared our model with STGCN, DCRNN, AGCRN and CLCRN. The reslut shows that our model outperform others especially in temperature prediction
![image](https://github.com/xzwbsz/DCPGRNcode/assets/44642002/38b6df9f-aa75-4e90-9f27-8bc514bd7456)



## Acknowledgement
The project is developed based on [PM2.5-GNN](https://github.com/shuowang-ai/PM2.5-GNN) and [IDGL](https://github.com/hugochan/IDGL) for dynamic graph.
