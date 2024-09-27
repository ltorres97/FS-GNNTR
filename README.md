## FS-GNNTR: Few-shot Learning with Transformers via Graph Embeddings for Molecular Property Prediction

In this work, we propose a few-shot GNN-Transformer architecture, FS-GNNTR to face the problem of low-data in molecular property prediction. It is demonstrated that this model provides strong boosts when predicting molecular properties on few-shot data over simple graph-based methods.

The GNN-Transformer network learns deep representations from graph-level embeddings. First, a GNN module encodes the structural information of molecular graphs as a set of node and edge features. Node and edge embeddings are then converted into graph embedding representations by neighborhood aggregation. Then, a vision Transformer encoder exploits the contextual information of these vectorial embeddings to propagate deep representations across attention layers.

![ScreenShot](FS-GNNTR/plots/architecture.png?raw=true)

A two-module meta-learning framework was explored to optimize model parameters across tasks and quickly adapt to new molecular properties on few-shot data. 

![ScreenShot](FS-GNNTR/plots/meta-fsgnntr.png?raw=true)

Extensive experiments on real multiproperty prediction data, Tox21 and SIDER, demonstrate the predictive power and stable performances of the proposed model when inferring task-specific molecular properties.

This repository provides the source code and datasets for the proposed work.

Article Link: https://doi.org/10.1016/j.eswa.2023.120005

Contact Information: (uc2015241578@student.uc.pt, luistorres@dei.uc.pt), if you have any questions about this work.

## Data Availability and Pre-Processing

The Tox21 and SIDER datasets are downloaded from the repository [Data](http://snap.stanford.edu/gnn-pretrain/data/) (chem_dataset.zip) from [Hu et al. (2020)](https://arxiv.org/abs/1905.12265). 

Raw data is pre-processed and SMILES strings are converted into molecular graphs using RDKit.Chem. 

The implementation is based on [Strategies for Pre-training Graph Neural Networks (Hu et al.) (2020)](https://arxiv.org/abs/1905.12265).

## Python Packages 

We used the following Python packages for core development. We tested on Python 3.7.

```
- torch = 1.10.1
- torch-cluster = 1.5.9
- torch-geometric = 2.0.4
- torch-scatter = 2.0.9
- torch-sparse = 0.6.12
- torch-spline-conv = 1.2.1
- torchvision = 0.10.0
- vit-pytorch = 0.35.8
- scikit-learn = 1.0.2
- seaborn = 0.11.2
- scipy = 1.8.0
- numpy = 1.21.5
- tqdm = 4.50.0
- tensorflow = 2.8.0
- keras = 2.8.0
- tsnecuda = 3.0.1
- tqdm = 4.62.3
- matplotlib = 3.5.1
- pandas = 1.4.1
- networkx = 2.7.1
- rdkit
```

## References

[1] Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J.: Strategies for pre-training graph neural networks. CoRR abs/1905.12265 (2020). https://doi.org/10.48550/ARXIV.1905.12265
```
@inproceedings{
hu2020pretraining,
title={Strategies for Pre-training Graph Neural Networks},
author={Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJlWWJSFDH},
}
```

[2] Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: 34th International Conference on Machine Learning, ICML 2017, vol. 3 (2017). https://doi.org/10.48550/arXiv.1703.03400
```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}

```

[3] Guo, Z., Zhang, C., Yu, W., Herr, J., Wiest, O., Jiang, M., & Chawla, N. V. (2021). Few-shot graph learning for molecular property prediction. In The Web Conference 2021 - Proceedings of the World Wide Web Conference, WWW 2021 (pp. 2559–2567). Association for Computing Machinery, Inc. https://doi.org/10.1145/3442381.3450112
```
@article{guo2021few,
  title={Few-Shot Graph Learning for Molecular Property Prediction},
  author={Guo, Zhichun and Zhang, Chuxu and Yu, Wenhao and Herr, John and Wiest, Olaf and Jiang, Meng and Chawla, Nitesh V},
  journal={arXiv preprint arXiv:2102.07916},
  year={2021}
}

```

[4] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. https://doi.org/10.48550/arxiv.2010.11929
```
@article{Dosovitskiy2020,
   author = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
   doi = {10.48550/arxiv.2010.11929},
   month = {10},
   title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
   url = {https://arxiv.org/abs/2010.11929},
   year = {2020},
}
```

[5] Vision Transformers with PyTorch. https://github.com/lucidrains/vit-pytorch
```
@misc{Phil Wang,
  author = {Phil Wang},
  title = {Vision Transformers},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lucidrains/vit-pytorch}},
}
```

