Eval on test set with epoch 10 weights used since best val accuracy:

Accuracy:  0.8210
Precision: 0.8729
Recall:    0.7513
F1 score:  0.8075
Confusion matrix:
[[14599  1792]
 [ 4073 12304]]

Reproduce with (from the repository root):

```bash
python eval.py --method method2 \
  --checkpoint method2/cp-epoch-10.weights.h5 \
  --batch-size 128 \
  --threshold 0.5
```
