import numpy as np
from datasets import Sequence, Value

#ds: datasets.Dataset Object
def preprocess(ds, image_col_name="image"):
    # Store each image as a row-major sequence of RGB pixels.
    def convert_to_float(example):
        image = example[image_col_name]
        image = image.convert("RGB")  # Ensure image is in RGB format
        image = np.array(image).astype(np.float32) / 255.0
        example[image_col_name] = image.reshape(-1, 3, order="C")
        return example

    features = ds.features.copy()
    features[image_col_name] = Sequence(
        Sequence(Value("float32"), length=3)
    )
    return ds.map(convert_to_float, batched=False, features=features)
