# Dialogue Act Classification

PyTorch implementation of Dialogue Act Classification using BERT and RNN with Attention. 

### Implementation details

BERT backbone is used to obtain utterance-level representations. BERT is fine-tuned for the first 3 epochs and then its weights are freezed.

GRU with Attention is used to process a sequence of utterances. Then, a linear layer with softmax is applied to get final predictions.

This implementation is partially based on:
- [Pierre Colombo, Emile Chapuis, Matteo Manica, Emmanuel Vignon, Giovanna Varni, Chloe Clavel, "Guiding attention in Sequence-to-sequence models for Dialogue Act prediction", arXiv:2002.08801](https://arxiv.org/abs/2002.08801)
- [Vipul Raheja, Joel Tetreault, "Dialogue Act Classification with Context-Aware Self-Attention", arXiv:1904.02594](https://arxiv.org/abs/1904.02594)

The main differences between this implementation and the papers:
- For utterance representations BERT contextualized embeddings are used instead of GloVe/ELMo/fastText with RNN.
- Vanilla softmax instead of CRF.
- Another type of Attention is used.

### Dataset

[The Switchboard Dialog Act Corpus (SwDA)](https://catalog.ldc.upenn.edu/LDC97S62) is used for training.

[swda GitHub repo](https://github.com/cgpotts/swda) is used to obtain the dataset.

Data is split into train, valid and test subsets according to ["Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks"](http://arxiv.org/abs/1603.03827) NAACL 2016 paper.

### Reproducing the results

1. Clone the repo: `git clone --recurse-submodules https://github.com/JandJane/DialogueActClassification.git`
2. Unzip data: `unzip DialogueActClassification/swda/swda.zip -d DialogueActClassification/swda/swda`
3. Install requirements: `pip install -r DialogueActClassification/requirements.txt`
4. Run `train_model.ipynb` notebook (training takes about 6 hours on Google Colab)

