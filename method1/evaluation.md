# Method 1 evaluation

## Evaluation setup

The selected Method 1 model is the epoch 7 checkpoint, `epoch-07.weights.h5`. It is ranked first in `ranking.csv` by the training run's model-selection rule: lowest validation loss, followed by highest validation accuracy and earliest epoch in the event of a tie. At epoch 7, its validation loss was 0.357594 and its validation accuracy was 0.860870.

The checkpoint was evaluated once on the complete official `test` split of [`1aurent/PatchCamelyon`](https://huggingface.co/datasets/1aurent/PatchCamelyon), with no shuffling or dropped final batch. The evaluation used all 32,768 test images and a fixed decision threshold of 0.5. Label 1 is treated as the positive (tumour) class and label 0 as the negative (non-tumour) class.

## Results

| Metric | Value |
|---|---:|
| Accuracy | **0.829926 (82.99%)** |
| Precision (positive predictive value) | **0.902294 (90.23%)** |
| Recall (sensitivity) | **0.739818 (73.98%)** |
| F1 score | **0.813018 (81.30%)** |
| Specificity | 0.919956 (92.00%) |
| Negative predictive value | 0.779679 (77.97%) |
| Balanced accuracy | 0.829887 (82.99%) |
| Matthews correlation coefficient | 0.670782 |
| False-positive rate | 0.080044 (8.00%) |
| False-negative rate | 0.260182 (26.02%) |

### Confusion matrix

Rows are actual labels and columns are predicted labels.

|  | Predicted negative (0) | Predicted positive (1) | Support |
|---|---:|---:|---:|
| **Actual negative (0)** | **15,079 (TN)** | **1,312 (FP)** | 16,391 |
| **Actual positive (1)** | **4,261 (FN)** | **12,116 (TP)** | 16,377 |
| **Total** | 19,340 | 13,428 | **32,768** |

## Interpretation

The model is highly precise when predicting tumour tissue (90.23%) and rejects non-tumour patches well (92.00% specificity). Its lower recall (73.98%) means that it misses 4,261 of the 16,377 positive test patches, a 26.02% false-negative rate. Consequently, the epoch-7 checkpoint favours precision over sensitivity at the default 0.5 threshold.

## Reproduction

Run from the repository root using the project's TensorFlow environment:

```bash
python model1_eval.py \
  --checkpoint method1/epoch-07.weights.h5 \
  --batch-size 128 \
  --threshold 0.5
```

Evaluation date: 7 October 2026.
