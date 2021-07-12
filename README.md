## ANGAN

- This repository is the implementation of:
Attribute Augmented Network Embedding Based on Generative Adversarial Nets


### Files in the folder
- `data/`: training data
- `pre_train/`: pre-trained node embeddings
  > Note: the dimension of pre-trained node embeddings should equal n_emb in src/GraphGAN/config.py
- `results/`: evaluation results and the learned embeddings of the generator
- `src/`: source codes


### Requirements
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):

- tensorflow == 1.8.0
- tqdm == 4.23.4 (for displaying the progress bar)
- numpy == 1.14.3
- sklearn == 0.19.1




