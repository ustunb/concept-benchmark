from dataclasses import dataclass
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from .utils import add_unobserved_rows_to_df
from .robots import Robot
from PIL import Image
from .paths import image_dir
import albumentations as A
import torch

@dataclass
class FeatureBinarizer(object):
    """Class to convert a categorical feature into a set binary variables"""
    name: str
    targets: list
    formatter: str = '%s_is_%s'
    def get_indicator_name(self, target = None, binary_valued = False):

        if binary_valued:
            return self.name

        if target is None:
            target = str(self.targets[0])

        assert target in self.targets
        return self.formatter % (self.name, target)

class DatasetBinarizer(object):
    """Class to store binarizers for all features in a ClassificationDataset"""

    def __init__(self, outcome_name, outcome_values, feature_names, feature_values):
        self._outcome_name = str(outcome_name)
        self._feature_names = [str(s) for s in feature_names]
        self._outcome_values = list(outcome_values)
        self._feature_values = {name: feature_values[name] for name in self._feature_names}
        self._bb = {k: FeatureBinarizer(name = k, targets = [v[0]]) for k, v in self._feature_values.items()}
        self._bb[self._outcome_name] = FeatureBinarizer(name = self._outcome_name, targets = [self._outcome_values[0]])
        self._outcome_map = {
            1: self._outcome_values[0],
            -1: self._outcome_values[1],
            }
        assert self._check_rep()

    @property
    def outcome_name(self):
        """name of the outcome variable (categorical)"""
        return self._outcome_name

    @property
    def outcome_indicator_name(self):
        """name of the outcome variable (binarized)"""
        return self._bb[self._outcome_name].get_indicator_name()

    @property
    def feature_names(self):
        """name of the features (categorical)"""
        return self._feature_names

    @property
    def feature_indicator_names(self):
        """name of the features (binarized)"""
        return [v.get_indicator_name() for k, v in self._bb.items() if k != self._outcome_name]

    def __eq__(self, other):
        if set(self._bb.keys()) != set(other._bb.keys()):
            return False

        for name, feature_binarizer in self._bb.items():
            if feature_binarizer != other._bb[name]:
                return False

        return True

    def copy(self):
        b = DatasetBinarizer(outcome_name = self._outcome_name, outcome_values = self._outcome_values, feature_names = list(self._feature_values.keys()), feature_values = dict(self._feature_values))
        b._bb = {name: FeatureBinarizer(name = binarizer.name, targets = binarizer.targets) for name, binarizer in self._bb.items()}
        return b

    def _check_rep(self):
        assert len(self._bb[self._outcome_name].targets) == 1, 'outcome variable should only have 1 target'
        for name, values in self._feature_values.items():
            assert len(self._bb[name].targets) < len(values)
        return True

    def apply(self, df, false_value = -1, return_complement_names = False):
        """
        convert dataframe with categorical columns into a dataframe with binary variables
        :param df:
        :return:
        """
        assert isinstance(df, pd.DataFrame)
        assert false_value in (0, -1)
        columns = [self._outcome_name] + list(self._feature_values.keys())
        columns = [name for name in columns if (name in self._bb) and (name in df.columns)]
        complement_names = dict()
        assert self._check_rep()

        bf = pd.DataFrame()
        for name in columns:

            values = df[name]
            categories = set(values)
            binarizer = self._bb[name]

            for target in binarizer.targets:
                binary_flag = values.isin((0, 1)).all()
                indicator_name = binarizer.get_indicator_name(target = target, binary_valued = binary_flag)
                bf[indicator_name] = values.isin([target])

                complements = categories - set([target])
                if len(complements) == 1:
                    complement_target = complements.pop()
                    complement_names[indicator_name] = binarizer.formatter % (name, complement_target)

        bf = bf.astype(int)
        bf.index = df.index
        if false_value != 0:
            bf = bf.replace(0, false_value)

        out = (bf,)
        if return_complement_names:
            out = (*out, complement_names)

        return out


    # def update(self, name, targets):
    #     """
    #     update the binarizer target list for a given feature
    #     :param name: name of the feature
    #     :param target: new target
    #     """
    #     assert isinstance(name, str)
    #     assert isinstance(targets, list)
    #
    #     valid_target_values = self._feature_values[name]
    #
    #     target_values = []
    #     for target in targets:
    #         if isinstance(target, str):
    #             assert target in valid_target_values
    #             target_value = target
    #         elif isinstance(target, int):
    #             assert target in range(len(valid_target_values))
    #             target_value = valid_target_values[target]
    #         else:
    #             raise ValueError()
    #         target_values.append(target_value)
    #
    #     self._bb[name] = FeatureBinarizer(name, targets = target_values)
    #     assert self._check_rep()

@dataclass
class ClassificationDataset(object):
    """
    Dataset for binary classification task. Here:

        df: data in (X, n_neg, n_pos) format
            each row of df corresponds to a labelled example (y, x[1], ... x[d])
            where y is a value of the outcome variable and x[1],...,x[d] are the values of
            d input variables. the outcome variable is always placed in the first column.
            all values in data_df are binary.

        df_cat: data in (X, y) format with categorical variables

        df_bin: data in (X, y) format with binary variables

        false_value: integer value of "False" points (either -1 or 0)
    """
    df_cat: pd.DataFrame
    binarizer: DatasetBinarizer

    def __post_init__(self):

        self._false_value = -1
        # names
        self.outcome_name = self.binarizer.outcome_name
        self.feature_names = self.binarizer.feature_names
        self.outcome_map = self.binarizer._outcome_map
        self.outcome_indicator_name = self.binarizer.outcome_indicator_name
        self.feature_indicator_names = self.binarizer.feature_indicator_names
        self.d = len(self.feature_indicator_names)

        # convert categorical dataset to binary dataset
        self.df_bin, complement_names = self.binarizer.apply(df = self.df_cat, false_value = self._false_value, return_complement_names = True)
        self.complement_names = complete_complements(complement_names)

        # store X, y matrices for training
        self.X = self.df_bin[self.feature_indicator_names].values
        self.y = self.df_bin[self.outcome_indicator_name].values
        self.Xy = self.df_bin[self.feature_indicator_names + [self.outcome_indicator_name]].values


        # convert binary dataset into standard "flat" form
        self.df = self._flatten_data_df(df = self.df_bin, false_value = self._false_value)
        self.counts = self.df.index.repeat(self.df['n_neg'] + self.df['n_pos']).values
        self.n_pos = self.df['n_pos'].sum()
        self.n_neg = self.df['n_neg'].sum()
        self.n = self.n_pos + self.n_neg

        # create dictionary of types
        self.bin_names = self.feature_indicator_names + [self.outcome_indicator_name]
        self.cat_names = self.feature_names + [self.outcome_name]
        self.df_cat_types = self.df_cat.drop_duplicates(ignore_index = False)[self.cat_names]
        self.df_bin_types = self.df_bin.drop_duplicates(ignore_index = False)[self.bin_names]

    def __add__(self, other):
        assert isinstance(other, ClassificationDataset)
        assert self.outcome_indicator_name == other.outcome_indicator_name
        assert set(self.feature_indicator_names) == set(other.feature_indicator_names)
        assert self.binarizer == other.binarizer
        return ClassificationDataset(df_cat = pd.concat((self.df_cat, other.df_cat)), binarizer = self.binarizer)

    @property
    def false_value(self):
        return self._false_value

    @false_value.setter
    def false_value(self, value):
        assert isinstance(value, (int, float)) and value in (-1, 0)
        value = int(value)
        if value != self._false_value:
            self.df[self.feature_indicator_names] = self.df[self.feature_indicator_names].replace(self._false_value, value)
            self.df_bin[self.feature_indicator_names] = self.df_bin[self.feature_indicator_names].replace(self._false_value, value)
            self.df_bin_types[self.feature_indicator_names] = self.df_bin_types[self.feature_indicator_names].replace(self._false_value, value)
            self._false_value = value

    def to_robots(self, predictor = None, add_cols=[], other_data=None):
        """
        converts dataset into a list of Robots
        :param predictor: model to assign predictions to each Robot
        :param add_cols: additional characteristics in the dataframe describing the robot
        :param other_data: pandas.DataFrame with other info about the robot
        :return: list of Robot objects
        """
        robots = []
        for idx, row in self.df_cat.iterrows():

            r = Robot(id = idx,
                      features=row[self.feature_names].to_dict(),
                      features_subtype={feat: other_data.loc[row.name][feat] for feat in self.feature_names} if other_data is not None else {},
                      type = row[self.outcome_name],
                      type_binary = self.df_bin.loc[idx][self.outcome_indicator_name],
                      x = self.df_bin[self.feature_indicator_names].loc[idx].to_numpy(),
                      kwargs = {k: int(other_data.loc[idx][k]) if (isinstance(other_data.loc[idx][k], (int, np.integer))
                                                                   or (isinstance(other_data.loc[idx][k], str)) and
                                                                   other_data.loc[idx][k].isdigit())
                                   else other_data.loc[idx][k] for k in add_cols})

            if predictor is not None:
                r.predictions.append(predictor.predict(X = r.x)[0])

            robots.append(r)

        return robots

    def get_index_from_numeric(self, values):
        """
        :param values: array-like containing x values
        :return: list of indices for matching values
        """
        values = np.atleast_2d(values)
        assert np.isin(values, (-1, 0, 1)).all()
        assert values.shape[1] in (self.d, self.d+1)
        np.place(values, values < 1, self.false_value)

        indices = []
        if values.shape[1] == self.d:
            for x in values:
                match = (self.df_bin_types[self.feature_indicator_names] == x).all(1)
                if any(match):
                    indices.append(self.df_bin_types.index[match][0])
        else:
            for xy in values:
                match = (self.df_bin_types == xy).all(1)
                if any(match):
                    indices.append(self.df_bin_types.index[match][0])

        return indices

    def get_index_from_categorical(self, values):
        """
        :param values: list of dictionaries of the form: {feature_name: feature_value}
                       elements can be objects that will be cast as dictionaries (e.g., pd.Series)
        :return: list of indices for matching values
        """

        assert isinstance(values, list)
        values = [dict(e) for e in values]
        values = [pd.Series(e) for e in values]
        indices = []
        for e in values:
            match = (self.df_cat_types == e).all(1)
            if any(match):
                indices.append(self.df_cat_types.index[match][0])

        return indices

    def _flatten_data_df(self, df, false_value = -1, complete = True):
        """
        converts a df of training data for a binary classification task with
        boolean attributes from (y, X) format into (X, n_neg, n_pos) format.

        Here, n_neg/n_pos is the number of examples in the training dataset with
        features X and a negative/positive label y = -1/+1, respectively.

        :param df: data.frame in (y, X) format
        :param false_value: integer value of "False" points (either -1 or 0)
        :param complete: set to true to add rows for unobserved values of x to the
                         flat df. The rows will be of the form (x, 0, 0), and
                         `flat_df` will contain exactly 2^d rows

        :return: data.frame in (X, n_neg, n_pos) format
        """
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] >= 2
        assert false_value in (0, -1)

        YX = np.array(self.df_bin[[self.outcome_indicator_name] + self.feature_indicator_names].values)
        assert np.isin(YX, (-1, 0, 1)).all()

        # split into Y and X
        Y_col_idx = 0
        X_col_idx = range(1, YX.shape[1])
        Y = YX[:, Y_col_idx].flatten()
        X = YX[:, X_col_idx]

        # overwrite false value
        np.place(X, X < 1, false_value)

        # distinct points
        pos_idx = Y == 1
        U_pos, n_pos = np.unique(X[pos_idx, :], axis = 0, return_counts = True)
        U_neg, n_neg = np.unique(X[~pos_idx, :], axis = 0, return_counts = True)

        # create data frame
        df_pos = pd.DataFrame(U_pos, columns = self.feature_indicator_names)
        df_pos['n_pos'] = n_pos
        df_neg = pd.DataFrame(U_neg, columns = self.feature_indicator_names)
        df_neg['n_neg'] = n_neg
        df_flat = pd.merge(df_neg, df_pos, how = 'outer', on = self.feature_indicator_names).fillna(0)

        # add unobserved rows
        if complete:
            df_flat = add_unobserved_rows_to_df(df_flat)
        else:
            df_flat = df_flat.astype(int).sort_values(by = self.feature_indicator_names)

        return df_flat

    def remove_duplicates(self):
        """
        remove duplicate rows from the dataset but keep the id
        :return: new ClassificationDataset object
        """
        df_cat = self.df_cat.drop_duplicates()
        return ClassificationDataset(df_cat = df_cat, binarizer = self.binarizer)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = [
            'ClassificationDataset<d = {} features, n = {} points>'.format(self.df_cat.shape[1], self.df_cat.shape[0]),
            'outcome (y = 1): {}'.format(self.binarizer.outcome_indicator_name),
            str(self.df_cat)
            ]
        return '\n'.join(s)

def complete_complements(a):
    """
    :param dictionary containing names of indicators and complements
            keys are the names of indicators
            values are complements
    :return: c: dictionary of indicators and complements that can be used for
            find and replace operations
            c = {~indicator_name: complement_name}
    """
    assert isinstance(a, dict)
    assert len(a) > 0
    c = {v: k for k, v in a.items()}
    c.update(a)
    c = {'~{}'.format(k): v for k, v in c.items()}
    return c


class RobotImageDataset(Dataset):

    def __init__(self, catalog, classification_dataset, processor, augment=False, copies_per_robot=1):
        """
        Args:
            classification_dataset: ClassificationDataset object
            processor: ViT image processor
            augment: Whether to apply data augmentation
            copies_per_robot: Number of image copies per robot for augmentation
        """
        self.classification_dataset = classification_dataset
        self.image_dir = image_dir
        self.processor = processor
        self.augment = augment
        self.copies_per_robot = copies_per_robot if augment else 1

        # Get the categorical dataframe
        self.df_cat = catalog

        # Get feature names (these are the visual concepts we want to predict)
        self.feature_names = classification_dataset.feature_names


        # Create expanded dataset with copies
        self.expanded_data = self._create_expanded_dataset()
        print(self.expanded_data.head().to_string())

        # Define augmentations
        if augment:
            self.aug_transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.Blur(blur_limit=5),
                ], p=0.3),
            ])
        else:
            self.aug_transform = None

    def _create_expanded_dataset(self):
        """
        Create expanded dataset by repeating each robot row multiple times
        """
        expanded_rows = []

        for copy_id in range(self.copies_per_robot):
            for robot_idx in self.df_cat.index:
                robot_row = self.df_cat.loc[robot_idx].copy()
                robot_row['copy_id'] = copy_id
                expanded_rows.append(robot_row)

        return pd.DataFrame(expanded_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.expanded_data)

    def __getitem__(self, idx):
        # Get robot data
        row = self.expanded_data.iloc[idx]
        image_path = self.image_dir / row['png_filename']

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)

            # Apply augmentations to copies
            if self.augment and self.aug_transform and row['copy_id'] > 0:
                augmented = self.aug_transform(image=image_array)
                image_array = augmented['image']
                image = Image.fromarray(image_array)

            # Process image with ViT processor
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension

            # Get feature labels from the categorical data
            # Convert categorical values to binary labels for the neural network
            feature_labels = []
            for feature in self.feature_names:
                feature_value = row[feature]

                # Convert categorical features to binary based on your data structure
                if feature == 'foot_shape':
                    # pointy = 1, flat = 0
                    feature_labels.append(1.0 if feature_value == 'pointy' else 0.0)
                elif feature == 'body_shape':
                    # round = 1, square = 0
                    feature_labels.append(1.0 if feature_value == 'round' else 0.0)
                elif feature == 'head_shape':
                    # round = 1, square = 0
                    feature_labels.append(1.0 if feature_value == 'round' else 0.0)
                elif feature == 'has_knees':
                    # true = 1, false = 0
                    feature_labels.append(1.0 if feature_value == 'true' else 0.0)
                else:
                    return ValueError(f"Unknown feature: {feature}")
            labels = torch.tensor(feature_labels, dtype=torch.float32)

            return pixel_values, labels

        except Exception as e:
            return ValueError(f"Error loading image {image_path}: {e}")

    def get_feature_names(self):
        """Return the feature names (visual concepts) being predicted"""
        return self.feature_names

    def get_sample_info(self, idx):
        """Get information about a specific sample"""
        row = self.expanded_data.iloc[idx]
        return {
            'id': row['id'],
            'copy_id': row['copy_id'],
            'png_filename': row['png_filename'],
            'features': {feature: row[feature] for feature in self.feature_names}
        }




