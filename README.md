
# Dialogue Act Classification

Implementation and comparison of several solutions for Dialogue Act Classification.

## Dataset

[The Switchboard Dialog Act Corpus (SwDA)](https://catalog.ldc.upenn.edu/LDC97S62) is used for training.

[swda GitHub repo](https://github.com/cgpotts/swda) is used to obtain the dataset.

Data is split into train, valid and test subsets according to ["Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks"](http://arxiv.org/abs/1603.03827) NAACL 2016 paper.

##  Results

| Model                                      | Accuracy, % |
|--------------------------------------------|-------------|
| Tf-Idf + LightGBM without context          | 63.89       |
| Fasttext + LightGBM without context        | 66.57       |
| Pretrained Bert + LightGBM without context | 66.61       |
| Fasttext + Hierarchical RNN                | 76.63       |
| Pretrained Bert + RNN                      | 76.56       |
| Fine-tuned Bert + RNN                      | **78.05**   |


## Reproducing the results

1. Clone the repo: `git clone --recurse-submodules https://github.com/JandJane/DialogueActClassification.git`
2. Unzip data: `unzip DialogueActClassification/swda/swda.zip -d DialogueActClassification/swda/swda`
3. Install requirements: `pip install -r DialogueActClassification/requirements.txt`
4. Run notebooks 01-07
