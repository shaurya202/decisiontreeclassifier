from collections import Counter

import random

class Node:
    """A node in the decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        """
        Create a decision tree node.

        PARAMETERS:
        - feature: int or None
            The index of the feature used for splitting at this node.
            Example: 0 = height, 1 = weight, etc.

        - threshold: float or int or None
            The value used to split the dataset.
            Example: if feature 2 <= 10, go left; else go right.

        - left: Node or None
            The left child node (represents "feature <= threshold").

        - right: Node or None
            The right child node (represents "feature > threshold").

        - label: int or None
            If this is a leaf node, this stores the predicted class.
            Example: 0 = Not Athletic, 1 = Athletic

        BEHAVIOR:
        - Internal nodes have feature + threshold + children
        - Leaf nodes have ONLY label (feature and threshold are None)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        """
        Determine whether this node is a leaf node.

        RETURNS:
        - True if this node has a label (meaning it is a final prediction)
        - False if this node still splits data using a feature and threshold

        Leaf nodes are used for final predictions in the tree traversal.
        """
        return self.label is not None


class DecisionTreeClassifier:
    """
    Simple Decision Tree Classifier using Gini Impurity.
    Supports numerical features and classification problems.
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features: int | None = None):
        """
        Initialize the decision tree.

        PARAMETERS:
        - max_depth: int or None
            Maximum depth the tree is allowed to grow.
            Prevents overfitting by limiting complexity.

        - min_samples_split: int
            Minimum number of samples required to attempt a split.

        STORES:
        - self.root: the root Node of the tree (built during training)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

        self.max_features = max_features

    def fit(self, X, y):
        """
        Train the decision tree on dataset.

        PARAMETERS:
        - X: list of lists (feature matrix)
            Each row is a sample.
            Each column is a feature.

        - y: list
            Class labels corresponding to each row in X.

        OUTPUT:
        - Stores the root node of the trained tree in self.root
        """
        self.root = self._build_tree(X, y, 0)

    def predict(self, X):
        """
        Predict class labels for a batch of inputs.

        PARAMETERS:
        - X: list of samples

        RETURNS:
        - list of predicted labels
        """
        return [self._predict(x, self.root) for x in X]

    def _predict(self, x, node):
        """
        Predict a SINGLE sample by traversing the tree.

        PARAMETERS:
        - x: one sample (list of features)
        - node: current node in the tree

        PROCESS:
        - While node is NOT a leaf:
            - Check feature value in x
            - Compare it to node.threshold
            - Move left or right accordingly
        - Once leaf is reached:
            - return node.label

        RETURNS:
        - predicted class label
        """
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        PARAMETERS:
        - X: dataset at current node
        - y: labels at current node
        - depth: current depth of the tree

        LOGIC CHECKS (in order):
        1. If all labels are identical:
            -> return a leaf node with that label

        2. If number of samples < min_samples_split:
            -> return leaf with majority class

        3. If max_depth reached:
            -> return leaf with majority class

        4. Find best split using _best_split

        5. If no valid split exists:
            -> return majority leaf

        OTHERWISE:
        6. Split data into (use best_split):
            - left subset (feature <= threshold)
            - right subset (feature > threshold)

        7. Recursively build:
            - left child subtree
            - right child subtree

        8. Return a Node storing:
            - feature
            - threshold
            - left child
            - right child
        """
        if len(set(y)) == 1:
            return Node(label=y[0])

        if len(X) < self.min_samples_split:
            return Node(label=self._majority(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(label=self._majority(y))

        best_split = self._best_split(X, y)

        if best_split is None:
            return Node(label=self._majority(y))

        best_index, best_threshold = best_split

        feat_ls = [feat[best_index] for feat in X]

        left_X, right_X = [], []
        left_y, right_y = [], []

        for index, individual_feat_and_feat in enumerate(zip(feat_ls, X)): # feat is horizontal, feat_ls is vertical
            individual_feat, feat = individual_feat_and_feat

            if individual_feat <= best_threshold:
                left_X.append(feat)
                left_y.append(y[index])
            else:
                right_X.append(feat)
                right_y.append(y[index])

        left_subtree = self._build_tree(left_X, left_y, depth+1)

        right_subtree = self._build_tree(right_X, right_y, depth+1)

        return Node(feature=best_index, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.

        OBJECTIVE:
        Choose split that maximizes information gain (reduces Gini impurity most).

        PROCESS:
        1. Compute current impurity of full dataset (parent Gini)
        2. For each feature:
            - Collect all unique values
            - Try each value as a threshold
        3. For each threshold:
            - Split dataset into left and right groups
            - Skip invalid splits (empty side)
            - Compute information gain
        4. Track best (feature, threshold) pair

        RETURNS:
        - (best_feature_index, best_threshold)
        OR
        - None if no useful split exists
        """
        full_impurity = self._gini(y)

        rows = len(X)
        cols = len(X[0])
        best_feat_and_threshold = [0, 0] # [best feature index, best threshold]
        best_gain = 0

        if self.max_features is None:
            candidate_features = range(cols)
        else:
            candidate_features = random.sample(
                range(cols),
                min(self.max_features, cols)
            )

        for j in candidate_features:
            unique_vals = set()
            for i in range(rows):
                unique_vals.add(X[i][j])

            unique_vals = sorted(unique_vals)
            for val in unique_vals:

                left, right = [], []
                for a, b in zip(X, y):
                    left.append(b) if a[j] <= val else right.append(b)

                if not (left and right):
                    continue

                info_gain = self._information_gain(full_impurity, left, right)

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feat_and_threshold = [j, val]

        return None if best_feat_and_threshold == [0, 0] else best_feat_and_threshold


    def _information_gain(self, parent, left, right):
        """
        Compute improvement from splitting dataset.

        FORMULA:
        IG = parent_gini
             - (len(left)/total)*gini(left)
             - (len(right)/total)*gini(right)

        PARAMETERS:
        - parent: Gini impurity of parent node
        - left: labels in left split
        - right: labels in right split

        RETURNS:
        - numeric information gain value
        """
        total = len(left) + len(right)
        info_gain = parent - ((len(left)*self._gini(left)/total) + (len(right)*self._gini(right)/total))
        return info_gain

    def _gini(self, labels):
        """
        Compute Gini impurity.

        FORMULA:
        Gini = 1 - sum(p_i^2)

        WHERE:
        - p_i = proportion of class i in dataset

        INTERPRETATION:
        - 0 = perfectly pure (all same class)
        - higher = more mixed classes

        PARAMETERS:
        - labels: list of class labels

        RETURNS:
        - float impurity score
        """
        counter_label = Counter(labels)

        amt_of_labels = len(labels)
        return 1 - sum([(count/amt_of_labels)**2 for count in counter_label.values()])

    def _majority(self, y):
        """
        Find most common class in a list.

        PARAMETERS:
        - y: list of labels

        RETURNS:
        - most frequent label (majority class)

        USED WHEN:
        - stopping early
        - no good split exists
        """
        return Counter(y).most_common(1)[0][0]

    def print_tree(self):
        self._print_node(self.root, 0)

    def _print_node(self, node, depth):
        indent = "  " * depth

        feature_names = [
            "Height (cm)",
            "Weight (kg)",
            "Age (years)",
            "Resting HR",
            "VO2 Max",
            "Weekly Exercise (hrs)",
            "Sprint Speed",
            "Body Fat %",
            "Strength (1-10)",
            "Sleep Hours"
        ]

        if node.is_leaf():
            label = "Athletic" if node.label == 1 else "Not Athletic"
            print(f"{indent}=> Predict: {label}")
            return

        feat_name = feature_names[node.feature]

        print(f"{indent}[{feat_name} <= {node.threshold}]")
        self._print_node(node.left, depth + 1)

        print(f"{indent}[{feat_name} > {node.threshold}]")
        self._print_node(node.right, depth + 1)

if __name__ == "__main__":
    X = []
    y = []

    for i in range(200):
        height = random.randint(150, 195)
        weight = random.randint(45, 100)
        age = random.randint(18, 45)

        resting_hr = random.randint(50, 90)
        vo2 = random.randint(30, 85)
        exercise = random.randint(0, 14)
        sprint = round(random.uniform(5.8, 9.5), 1)
        bodyfat = random.randint(8, 30)
        strength = random.randint(2, 10)
        sleep = random.randint(4, 9)

        score = 0

        if vo2 > 60: score += 2
        if exercise > 6: score += 2
        if resting_hr < 65: score += 1

        if strength > 7: score += 2
        if bodyfat < 18: score += 1

        if bodyfat > 25: score -= 2
        if resting_hr > 80: score -= 2
        if exercise < 2: score -= 2

        score += random.choice([-1, 0, 1])

        label = 1 if score >= 2 else 0

        X.append([
            height, weight, age,
            resting_hr, vo2, exercise,
            sprint, bodyfat, strength, sleep
        ])
        y.append(label)

    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X, y)

    test = [
        [151, 46, 18, 79, 41, 2, 6.4, 19, 4, 6],
        [166, 61, 24, 67, 51, 6, 7.7, 14, 6, 8],
        [171, 66, 26, 61, 57, 8, 8.2, 11, 8, 8],
        [176, 73, 31, 55, 63, 10, 8.8, 10, 9, 8],
        [184, 83, 36, 49, 71, 9, 9.2, 10, 9, 7],
        [169, 69, 42, 83, 36, 2, 6.1, 24, 3, 5],
        [159, 52, 20, 73, 47, 3, 6.9, 18, 5, 7],
        [189, 92, 39, 45, 75, 7, 9.1, 13, 8, 6]
    ]

    predictions = tree.predict(test)

    print("Predictions:")
    for sample, prediction in zip(test, predictions):
        label = "Athletic" if prediction == 1 else "Not Athletic"
        print(f"{sample} -> {label}")

    tree.print_tree()