# PCam training and evaluation

Train either model through the shared pipeline:

```bash
python train.py --method method1
python train.py --method method2
```

These commands use the defaults: `--epochs 10`, `--batch-size 64`, and `--seed 42`.

Method 1 uses the CNN in `simple_model.py`. Method 2 uses the frozen,
ImageNet-pretrained DenseNet121 in `densenet_model.py` and trains its binary
classification head. The first Method 2 run may download ImageNet weights.
Training requires a TensorFlow GPU, matching the previous Method 1 pipeline.

Both methods save timestamped runs with an `epoch-XX.weights.h5` checkpoint
for every epoch, `history.csv`, `ranking.csv`, `training.log`, and `run.json`.
The default output roots are `training_1/runs` and `training_2/runs`, respectively.
Run metadata records the method, settings, validation ranking, and checkpoint
verification results. Epochs are ranked by validation loss, then validation
accuracy, then epoch number.

Use `--output-root` to choose a different location, and `--epochs`,
`--batch-size`, and `--seed` to configure training. For a short smoke run:

```bash
python train.py --method method2 --epochs 1 --steps-per-epoch 2 --validation-steps 1
```

Use `python train.py --help` for all training options.

## Evaluation

Evaluate either model on every example in the official PCam test split:

```bash
python eval.py --method method1
python eval.py --method method2
```

These commands use `--batch-size 128` and `--threshold 0.5`. Their default
checkpoints are `method1/epoch-07.weights.h5` and
`method2/cp-epoch-10.weights.h5`, respectively, matching the existing evaluation
results. Evaluation reports accuracy, precision, recall, F1, and confusion counts.

To evaluate a checkpoint from a new training run or change inference options:

```bash
python eval.py --method method2 \
  --checkpoint "training_2/runs/<run-directory>/epoch-01.weights.h5" \
  --batch-size 64 --threshold 0.6
```

Replace `<run-directory>` with the timestamped directory printed by `train.py`.
Evaluation restores all weights from the checkpoint without downloading ImageNet
weights and can run on CPU. Use `python eval.py --help` for all evaluation options.
