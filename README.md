# concept-benchmark

## ConceptDataset Object

Creating a dataset object and cross-validation:

```python
import numpy as np
from concept_benchmark.data import ConceptDataset

# Size of dataset: (n samples, d features, c concepts)
n, d, c = 100_000, 1_000, 10

# Random example
X = np.random.randn(n, d)
C = np.random.randn(n, c)
y = np.random.randint(0, 2, n)

# Metadata related to the dataset
meta = dict(
    data_type="image",                         # in ["image", "timeseries", "text", "tabular"]
    classes=["class0", "class1"],              # class names
    concepts=["concept_{i}" for i in range(c)] # concept names
)

# Concstruct dataset object
data = ConceptDataset(X, C, y, meta)

# Generate CV folds
data.generate_cvindices(seed=0)
data.split("K05N01", fold_num_validation=1, fold_num_test=2)

data.training.X, data.training.C, data.training.y       # access training set
data.validation.X, data.validation.C, data.validation.y # access validation set
data.test.X, data.test.C, data.test.y                   # access validation set
```

As you build your dataset, it may need custom methods.

```python
class CustomConceptDataset(ConceptDataset):

    def __init__(self, X, C, y, meta):
        super().__init__(X, C, y, meta)
        # set additional attributes need for your use case

    def custom_method(self):
        pass  # do additional stuff

dataset = CustomConceptDataset(X, C, y, meta)
```

## Creating Dataset

The idea here is to have a function that returns ConceptDataset with X, C, y defined.

Add a py files to: concept_benchmark/synthetic/your_dataset_name.py

```python
def create_synthetic_dataset(*args, **kwargs): # name your args and kwargs
    # ... do stuff to define X, C, y
    return CustomConceptDataset(X, C, y, meta)
```

## Contributions

Once you have a valid function in concept_benchmark/synthetic/{dataset_name}.py,
make an pull request on github and @ryanhammonds for review.


## Axes of Variation

- Robot
    - concepts: 4 visual features defining robot appearance
    - Binary: head_shape, body_shape, has_knees, foot_shape
    - num_robots: Unique robot designs using concepts (where max is: 16 concept-unique * 3 subtypes of foot_shape * 108 color color schemes)
    - samples_per_instance: 2: Color variations per robot
    - model: Labeling function mapping concepts to glorp/drent classification
    - model_type: 'deterministic' or 'stochastic' concept-to-label mapping
    - resolution: Generated image size resolution x resolution (images are square)
    - color_mode: 'color' or 'greyscale' for whether the image is colored or... in greyscale

- Sudoku
    - Size of board
    - Label/concept noise
    - Invalid board
    - Complete
    - Consistent
    - Valid

- Robot Text
    - Shapes
    - Sizes
    - Curvature
    - Subjective
    - Color