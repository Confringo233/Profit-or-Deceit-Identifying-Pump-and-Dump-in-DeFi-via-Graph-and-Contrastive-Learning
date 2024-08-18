<<<<<<< HEAD
# Profit or Deceit? Identifying Pump and Dump in DeFi via Graph and Contrastive Learning

## Introduction

The code for the paper "Profit or Deceit? Identifying Pump and Dump in DeFi via Graph and Contrastive Learning" combines graph neural networks with temporal relationships. In our work, we propose a Pump and Dump detection model based on T2BG (Temporal Transaction-based Behavior Graph) and name it PumpPatrol. The core idea is to integrate graph neural networks with transaction time information to effectively capture and predict the evolution of edges in graph data, thereby timely detecting the sudden occurrence of Pump and Dump events in a temporal sequence.

## Running the experiments

### Requirements

```
python==3.9.19
cuda==12.2
torch==1.12.0+cu113
torchvision==0.13.0+cu113
dgl==0.9.1
lightgbm==4.3.0
scikit-learn==1.3.2
scipy==1.12.0
```

### Dataset

Due to github's file size limitation, it is necessary to first merge the split dataset into a `pd.csv` file using the following command:

```
python data_merge.py
```

### Model training

Use the following command to train the model:

```
python train.py
```

### Optional arguments

During training, you can specify the following parameters:

```
--epochs          epochs for training on entire dataset
--batch_size      size of each batch
--embedding_dim   embedding dim for link prediction
--memory_dim      dimension of memory
--temporal_dim    temporal dimension for time encoding
--n_neighbors     number of neighbors while doing embedding
--num_heads       number of heads for multihead attention mechanism
--k_hop           sampling k-hop neighborhood
```

=======
# Profit-or-Deceit-Identifying-Pump-and-Dump-in-DeFi-via-Graph-and-Contrastive-Learning
>>>>>>> 4a9ecd1779671e9288ab54222f8f8561ecdfdb6e
