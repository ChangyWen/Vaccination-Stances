## Silent voices, visible patterns: Leveraging partial social interaction graph to detect vaccine hesitancy

### Terminologies
##### Methods:
  * `GNN-SD`: Our proposed method. A graph neural network (GNN) based model for solving the task of stance detection, which uses two separate sub-models for message passing along the in-edges and out-edges, respectively.
  * `BoT`: Bag-of-Tricks (Wang et al. 2021). BoT is a graph attention network-based method with various effective tricks imposed. It is one of the state-of-the-art GNN-based node classification methods according to the open graph benchmark leaderboard.
  * `NGNN`: Network in GNN (Song et al. 2021). NGNN is built based on BoT, which deepens the model of BoT by embedding two non-linear feed forward neural network layers into each graph convolutional layer. It is another state-of-the-art GNN-based node classification method according to the open graph benchmark leaderboard.
  * `LPA`: Label Propagation Algorithm (Zhu 2005). LPA is a classic and well-known semi-supervised learning technique. LPA assumes that any two nodes connected by an edge in the graph are likely to have the same label. The main intuition of LPA is to predict unlabeled nodes by propagating the observed labels across the edges of the graph.
##### Datasets:
  * `C-VS`: Our constructed dataset that consists of an interaction graph w.r.t. COVID-19 vaccination stances.

### Data Preprocessing
  * Pre-process the dataset `C-VS` via `src/twitter_func.py` and `src/twitter_dataset.py`.

### To run the evaluation
  * Evaluate various methods (except `LPA`) on `C-VS` via `src/twitter.py`.
  * Evaluate `LPA` on `C-VS` via `src/lpa_twitter.py`.
  * Please see the optional arguments by running the scripts with `-h` flag.

### Dataset Download
  * Please download our proposed dataset `C-VS` at the following anonymous Google Drive URL:
    * https://drive.google.com/drive/folders/1YRjSxrP39mLdMenCukgirSLQI0M-dbwr?usp=sharing

### References
- Song, X.; Ma, R.; Li, J.; Zhang, M.; and Wipf, D. P. 2021. Network In Graph Neural Network. arXiv preprint arXiv:2111.11638.
- Wang, Y.; Jin, J.; Zhang, W.; Yu, Y.; Zhang, Z.; and Wipf, D. 2021. Bag of tricks for node classification with graph neural networks. In Proc. of DLG-KDD.
- Zhu, X. J. 2005. Semi-supervised learning literature survey. Technical Report 1530, University of Wisconsin-Madison Department of Computer Sciences.