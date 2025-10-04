import numpy as np
import pandas as pd
from utils import DEFAULT_ROBOT_SETTINGS

from sklearn.model_selection import train_test_split
from concept_benchmark.models import FrontEndModel
from concept_benchmark.synthetic.robot import create_synthetic_dataset

# NOTE: this script doesn't use the prescribed labeling model
settings = DEFAULT_ROBOT_SETTINGS.copy()

# Generate dataset without collapsing subtypes (i.e., for hands and feet)
settings['collapse'] = False
data = create_synthetic_dataset(**settings)

# Generate dataset with collapsing subtypes
settings['collapse'] = True
col_data = create_synthetic_dataset(**settings)

df = pd.DataFrame(data.C, columns=data.meta['concepts'])
col_df = pd.DataFrame(col_data.C, columns=col_data.meta['concepts'])

# Create skewed dataset
# All flat feet
flat1 = df[df['foot_shape_flat_4sided'] == 1]
flat2 = df[df['foot_shape_flat_5sided'] == 1]
flat3 = df[df['foot_shape_flat_lshaped'] == 1]

# sample and assign y with different probabilities
# The ratios add up to 4 : 16 = 1 : 4 --> 20% positive rate overall
flat1_samp = flat1.sample(n=90, replace=False).assign(y_prob=0.5/8.5)  # 0.5 : 8.0
flat2_samp = flat2.sample(n=70, replace=False).assign(y_prob=0.5/6.5)  # 0.5 : 6.0
flat3_samp = flat3.sample(n=40, replace=False).assign(y_prob=3/5)  # 3 : 2
flat1_samp['y'] = flat1_samp['y_prob'].apply(lambda x: np.random.random() < x).astype(int)
flat2_samp['y'] = flat2_samp['y_prob'].apply(lambda x: np.random.random() < x).astype(int)
flat3_samp['y'] = flat3_samp['y_prob'].apply(lambda x: np.random.random() < x).astype(int)

flat_df = pd.concat([flat1_samp, flat2_samp, flat3_samp])
# check positive rate (may differ from 0.2 slightly due to randomness)
print(flat_df['y'].mean())

# pointy feet -> always positive
pointy_samp = df[
    (df['foot_shape_pointy_3sided'] == 1) |
    (df['foot_shape_pointy_4sided'] == 1) |
    (df['foot_shape_pointy_6sided'] == 1)
    ].sample(n=200, replace=False).assign(y=1)


x_df = pd.concat([pointy_samp, flat1_samp, flat2_samp, flat3_samp]).drop(columns=['y_prob'])
X = x_df.iloc[:, :-1].values
y = x_df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fe = FrontEndModel()
fe.fit(X_train, y_train)

print(f"training acc: {(fe.predict(X_train) == y_train).mean()}")
print(f"test acc: {(fe.predict(X_test) == y_test).mean()}")

# Evaluate on a larger test set with different distribution
test_samp = df.sample(n=10000, replace=True)
test_samp['y_prob'] = test_samp.apply(lambda x: 0.2 if x.iloc[:3].sum() == 1 else 1, axis=1)
test_samp['y'] = test_samp['y_prob'].apply(lambda x: np.random.random() < x).astype(int)

print(f"deployment set acc: {(fe.predict(test_samp.iloc[:, :-2].values) == test_samp['y']).mean()}")