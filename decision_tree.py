import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
col_names = ['id','age','job','material','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
data = pd.read_csv("data.csv", skiprows=1, header=None, names=col_names)

# Data preprocessing
data = data.drop(['contact', 'pdays', 'poutcome'], axis=1)
data = data[data['age'] > 0]
data = data.drop(['previous', 'id'], axis=1)

# Balance the dataset
label_counts = data['y'].value_counts()
num_yes = label_counts['yes']
num_no = label_counts['no']

if num_no > 2 * num_yes:
    no_samples_to_keep = 2 * num_yes
    no_indices_to_keep = data[data['y'] == 'no'].sample(n=no_samples_to_keep, random_state=42).index
    data_balanced = data.loc[no_indices_to_keep.union(data[data['y'] == 'yes'].index)]
else:
    data_balanced = data

# Shuffle the dataset
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Feature engineering
X = data_balanced.iloc[:, :-1]
Y = data_balanced.iloc[:, -1].values.reshape(-1, 1)
continuous_columns = ['age', 'balance', 'day', 'duration']
num_bins = 5
categorical_columns = ['c_age', 'c_balance', 'c_day', 'c_duration']

for i, col in enumerate(continuous_columns):
    bins = pd.qcut(X[col], num_bins, labels=False, duplicates='drop')
    X[categorical_columns[i]] = pd.Categorical(bins, categories=range(num_bins))
X.drop(continuous_columns, axis=1, inplace=True)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=.2, random_state=41)

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def Build(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.SplitChoice(dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.Build(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.Build(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.Leaf(Y)
        return Node(value=leaf_value)
    
    def SplitChoice(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.InfoGain(y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def InfoGain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.Gini(parent) - (weight_l * self.Gini(l_child) + weight_r * self.Gini(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def Gini(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def Leaf(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def ShowTree(self, tree=None, indent="-"):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "=?", tree.threshold)
            print("%sleft:" % (indent), end="")
            self.ShowTree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.ShowTree(tree.right, indent + indent)
    
    def Fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.Build(dataset)
    
    def Predict(self, X):
        return [self.MakePrediction(x, self.root) for x in X]
    
    def MakePrediction(self, x, tree):
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.MakePrediction(x, tree.left)
        else:
            return self.MakePrediction(x, tree.right)



#MSE Computing

def mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must be the same.")
    
    Y_test_df = pd.DataFrame(y_true, columns=['y'])
    Y_pred_df = pd.DataFrame(y_pred, columns=['y'])

    label_map = {"yes": 1, "no": 0}

    Y_test_df['y'] = Y_test_df['y'].map(label_map)
    Y_pred_df['y'] = Y_pred_df['y'].map(label_map)

    squared_errors = [(true - pred) ** 2 for true, pred in zip(Y_test_df['y'], Y_pred_df['y'])]

    mse = sum(squared_errors) / len(y_true)

    return mse



# Example usage
DT = DecisionTree(min_samples_split=3, max_depth=3)
DT.Fit(X_train, Y_train)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error (MSE):", mse)
print('Accuracy  :',1-mse)

DT.ShowTree()
Y_pred = DT.predict(X_test)



