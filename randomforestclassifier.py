from collections import Counter
import random
from decision_tree import DecisionTreeClassifier



class RandomForestClassifier:

    def __init__(
        self,
        n_trees=20,
        max_depth=None,
        min_samples_split=2,
        max_features=None
    ):
        # Store how many decision trees this forest should contain
        self.n_trees = n_trees

        # Store the maximum depth each decision tree is allowed to grow
        self.max_depth = max_depth

        # Store the minimum number of samples required to split a node
        self.min_samples_split = min_samples_split

        # Store how many random features each tree should consider
        # None means use all available features
        self.max_features = max_features

        # This list will hold all of the trained decision trees
        self.trees = []


    def fit(self, X, y) -> None:
        """
        Train the Random Forest.

        Steps:
        1. Clear any old trees that might already exist.
        2. Repeat n_trees times:
            a. Create a bootstrap sample of the training data.
            b. Create a new DecisionTreeClassifier.
            c. Train the tree on the bootstrap sample.
            d. Add the trained tree to self.trees.

        Remember:
        A Random Forest is just many decision trees combined together.
        """

        self.trees = []

        for _ in range(self.n_trees):
            bootstrap_X, bootstrap_y = self._bootstrap_sample(X, y)
            decision_tree = DecisionTreeClassifier(self.max_depth, self.min_samples_split,
                                                   self.max_features)
            decision_tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(decision_tree)


    def predict(self, X):
        """
        Predict labels for multiple samples.

        Steps:
        1. Loop through every sample in X.
        2. Call _predict(sample) to get the forest's prediction.
        3. Return a list containing all predictions.

        Example:
        [
            [height, weight, age],
            [height, weight, age]
        ]

        becomes:

        [
            0,
            1
        ]
        """

        return [self._predict(sample) for sample in X]


    def _predict(self, sample: list[int]) -> int:
        """
        Predict a single sample.

        Steps:
        1. Create an empty list to store votes from trees.
        2. Loop through every decision tree in the forest.
        3. Ask each tree for its prediction.
        4. Store each tree's prediction.
        5. Return the majority vote.

        Example:

        Tree 1 predicts: 1
        Tree 2 predicts: 0
        Tree 3 predicts: 1

        Final prediction:
        1
        """
        votes = [tree.predict([sample])[0] for tree in self.trees]

        return self._majority_vote(votes)

    def _bootstrap_sample(self, X, y) -> tuple[list[list[int]], list[int]]:
        """
        Create a bootstrap dataset.

        A bootstrap sample:
        - Has the same size as the original dataset.
        - Is created by randomly selecting examples.
        - Allows duplicates.
        - Does NOT remove examples from the original dataset.

        Example:

        Original:

        A B C D E

        Possible bootstrap sample:

        A A C E E

        Steps:
        1. Create empty lists for the new X and y data.
        2. Repeat len(X) times:
            a. Pick a random index from the dataset.
            b. Add X[index] to the new feature list.

            c. Add y[index] to the new label list.
        3. Return the new X and y lists.

        """
        new_X, new_y = [], []

        len_x = len(X)

        for _ in range(len_x):
            rand_index = random.randint(0, len_x - 1)
            new_X.append(X[rand_index])
            new_y.append(y[rand_index])

        return (new_X, new_y)


    def _majority_vote(self, predictions):
        """
        Decide the final prediction from all tree predictions.

        Example:

        predictions = [
            1,
            1,
            0,
            1,
            0
        ]

        The answer should be:

        1

        because it appears the most times.

        Hint:
        You already imported Counter.
        """

        return Counter(predictions).most_common(1)[0][0]


    def print_forest(self):
        """
        Print every decision tree in the forest.

        Steps:
        1. Loop through every tree.
        2. Print the tree number.
        3. Call that tree's print_tree() method.

        Example output:

        ----------------
        Tree 1
        ----------------

        [Feature <= threshold]
            Predict: 0

        ----------------
        Tree 2
        ----------------

        ...
        """

        dashes = "----------------"
        for index, tree in enumerate(self.trees):
            print(f"{dashes}\nTree {index+1}\n{dashes}\n")
            tree.print_tree()
            print("\n")
        