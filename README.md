# FSGNN
Implementation of FSGNN. For more details, please refer to [our paper](https://arxiv.org/abs/2105.07634)
Experiments were conducted with following setup:  
Pytorch: 1.6.0  
Python: 3.8.5  
Cuda: 10.2.89
Trained on NVIDIA V100 GPU.

**Summary of results**

| **Dataset** | **3-hop Accuracy(%)** | **8-hop Accuracy(%)** | **16-hop Accuracy(%)** | **32-hop Accuracy(%)** |
| :-------------- | :---------------------: | :---------------------: | :--------------------: | :--------------------: |
| Cora            | 87\.73                  |       **87\.93**        | 87\.91                 | 87\.83                 |
| Citeseer        | 77\.19                  | 77\.40                  | **77\.46**             | **77\.46**             |
| Pubmed          | 89\.73                  |       **89\.75**        | 89\.60                 | 89\.63                 |
| Chameleon       | 78\.14                  | 78\.27                  | 78\.36                 | **78\.53**             |
| Wisconsin       |       **88\.43**        | 87\.84                  | 88\.04                 | 88\.24                 |
| Texas           |       **87\.30**   |       **87\.30**        | 86\.76                 | 86\.76                 |
| Cornell         | 87\.03                  | 87\.84                  | 86\.76                 | **88\.11**             |
| Squirrel        | 73\.48                  | 74\.10                  | 73\.95                 | **74\.15**             |
| Actor           | 35\.67                  |       **35\.75**        | 35\.25                 | 35\.22                 |
| Actor(no-norm)  | 37\.63                  | **37\.75**              | 37\.67                 | 37\.62                 |

**To run node classification on different hops:**

**3-hop** : ```./run_classification_3_hop.sh```

**8-hop** : ```./run_classification_8_hop.sh```

**16-hop** : ```./run_classification_16_hop.sh```

**32-hop** : ```./run_classification_32_hop.sh```


In addition, we include model accuracy of Actor dataset without using hop-normalization, as model shows higher accuracy in this setting.




(Results may vary slightly with a different platform, e.g. use of different GPU. In such case, for best performance, some hyperparameter search may be required. Please refer to [PyTorch documentation](https://pytorch.org/docs/stable/notes/randomness.html) for more details.)


