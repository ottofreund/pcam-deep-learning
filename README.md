# PCam training

Train either model through the shared pipeline:

```bash
python train.py --method method1
python train.py --method method2
```

These commands use the defaults: `--epochs 10`, `--batch-size 64`, and `--seed 42`.

Method 1 uses the CNN in `pcam_model.py`. Method 2 uses the frozen,
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

Use `python train.py --help` for all options. Existing saved checkpoints in
`method1/` and `method2/` remain usable with the evaluation scripts.
