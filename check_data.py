from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_dataset

DATASET = "1aurent/PatchCamelyon"
# The first run downloads the dataset and stores it in the Hugging Face cache.
ds = load_dataset(DATASET)
# Check split sizes and binary-label balance.
for split in ["train", "valid", "test"]:
    labels = ds[split]["label"]
    counts = Counter(labels)
    print(
        split,
        "n =", len(ds[split]),
        "negative =", counts[False],
        "positive =", counts[True],
    )
# Check one image.
example = ds["train"][0]
image = example["image"]
#print first pixel of first image
print("first pixel of first image:", image.getpixel((0, 0))) # expected: (0, 0, 0)

print("image size:", image.size) # expected: (96, 96)
print("image mode:", image.mode) # expected: RGB
print("label:", bool(example["label"]))
# Display four examples from each class.
examples = {False: [], True: []}
for row in ds["train"]:
    y = bool(row["label"])
    if len(examples[y]) < 4:
        examples[y].append(row["image"])
    if len(examples[False]) == 4 and len(examples[True]) == 4:
        break
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for r, y in enumerate([False, True]):
    for c, img in enumerate(examples[y]):
        axes[r, c].imshow(img)
        axes[r, c].set_title("positive" if y else "negative")
        axes[r, c].axis("off")
plt.tight_layout()
plt.show()
# Expected documented counts: train 262,144; valid 32,768; test 32,768; every split 50% negative and 50% positive [2].