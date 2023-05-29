Code and data for paper Prototypical Networks Relation Classification Model Based on Entity Convolution.

---
dataset from [FewRel](https://thunlp.github.io/1/fewrel1.html)

---
Simply train or test a model with python main.py --train/test. Other parameters can be specified:

```bash
$ python main.py -h
usage: main.py [-h] [--dataset DATASET] [--trainN TRAINN] [--N N] [--K K]
               [--Q Q] [--batch_size BATCH_SIZE] [--train_iter TRAIN_ITER]
               [--val_iter VAL_ITER] [--test_iter TEST_ITER]
               [--eval_every EVAL_EVERY] [--max_length MAX_LENGTH] [--lr LR]
               [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT]
               [--na_rate NA_RATE] [--grad_iter GRAD_ITER]
               [--hidden_size HIDDEN_SIZE] [--load_model LOAD_MODEL] [--train]
               [--test] [--dot] [--conv_num] [--conv_size]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     name of dataset
  --trainN TRAINN       N in train
  --N N                 N way
  --K K                 K shot
  --Q Q                 Num of query per class
  --batch_size BATCH_SIZE
                        batch size
  --train_iter TRAIN_ITER
                        num of iters in training
  --val_iter VAL_ITER   num of iters in validation
  --test_iter TEST_ITER
                        num of iters in testing
  --eval_every EVAL_EVERY
                        evaluate after training how many iters
  --max_length MAX_LENGTH
                        max length
  --lr LR               learning rate
  --weight_decay WEIGHT_DECAY
                        weight decay
  --dropout DROPOUT     dropout rate
  --na_rate NA_RATE     NA rate (NA = Q * na_rate)
  --grad_iter GRAD_ITER
                        accumulate gradient every x iterations
  --hidden_size HIDDEN_SIZE
                        hidden size
  --load_model LOAD_MODEL
                        where to load trained model
  --train               whether do training
  --test                whether do testing
  --dot                 use dot instead of L2 distance for proto
  --conv_num            the number of convolutions
  --conv_size           the size of convolutions
```
our result in FewRel test set.
| model | 5-way 1-shot |5-way 5-shot |10-way 1-shot |10-way 5-shot |
|--|--|--|--|--|
| Meta Network           | 64.46   | 80.57  | 53.96  | 69.23  |
| GNN                    | 66.23   | 81.28  | 46.27  | 64.02  |
| SNAIL                  | 67.29   | 79.40  | 53.28  | 68.33  |
| Proto                  | 69.20   | 84.79  | 56.44  | 75.55  |
| Proto-HATT             | -       | 90.12  | -      | 83.05  |
| HAPN                   | -       | 88.45  | -      | 80.26  |
| MLMAN                  | 82.08   | 92.66  | 75.59  | 87.29  |
| BERT-PAIR              | **88.32** | 93.22  | **80.63** | 87.02  |
| Ensemble               | 81.35   | 92.90  | 71.29  | 87.30  |
| Proto-Mul+AT           | -       | 90.70  | -      | 83.70  |
| ECRC(ours)             | 87.45   | **94.48** | 78.25  | **90.65** |

our result in FewRel 2.0 set.
| model | 5-way 1-shot |5-way 5-shot |10-way 1-shot |10-way 5-shot |
|--|--|--|--|--|
| proto-CNN             | 35.09  | 49.37  | 22.98  | 35.22  |
| proto-bert            | 40.12  | 51.50  | 26.45  | 36.93  |
| BERT-PAIR             | 56.25  | 67.44  | 43.64  | 53.17  |
| Ensemble              | -      | 68.32  | -      | 63.98  |
| DaFeC                 | 61.20  | 76.99  | 47.63  | 64.79  |
| ECRC(ours)            | **67.41** | **85.15** | **52.22** | **76.34** |



**Prototypical networks relation classification model based on entity convolution**


link:[https://www.sciencedirect.com/science/article/pii/S0885230822000602](https://www.sciencedirect.com/science/article/pii/S0885230822000602)


```bash
@article{feng2023prototypical,
  title={Prototypical networks relation classification model based on entity convolution},
  author={Feng, Jianzhou and Wei, Qikai and Cui, Jinman},
  journal={Computer Speech \& Language},
  volume={77},
  pages={101432},
  year={2023},
  publisher={Elsevier}
}
```

If you have any questions, please contact mrweiqk@163.com
