from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForestClassifierCustom:
    def __init__(self, n_classes, **kwargs):
        # Initialize the random forest classifier with any given arguments
        self.model = RandomForestClassifier(**kwargs)
        self.n_classes = n_classes
        self.n_estimators = self.model.n_estimators

    @property
    def estimators_(self):
        # Access the estimators from the internal RandomForestClassifier
        return self.model.estimators_ if hasattr(self.model, 'estimators_') else None

    def fit(self, X, y):
        """
        Trains the model.
        """
        self.classes_trained_on = np.sort(np.unique(y))

        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Predicts probabilities on the test set and expands output to n_classes dimensions.
        N = n_samples
        c = n_classes_trained_on
        Cl = tot n_classes
        """
        # Get predictions only for classes observed at fit
        probs = self.model.predict_proba(X)    # [N, c]
        
        # Initialize a matrix of zeros with shape [N, Cl]
        expanded_probs = np.zeros((X.shape[0], self.n_classes))
        
        # Assign probabilities to the correct class indices
        for idx, c in enumerate(self.classes_trained_on):
            expanded_probs[:, c] = probs[:, idx]
        
        return expanded_probs


# if __name__ == "__main__":
#     n_train_samples = 10
#     n_test_samples = 100
#     n_features = 4
#     classes_train = np.array([2, 9])
#     n_classes = 10
#     X_train = np.random.rand(n_train_samples, n_features)
#     y_train = np.random.choice(classes_train, size=n_train_samples, replace=True)

#     X_test = np.random.rand(n_test_samples, n_features)
#     y_test = np.random.choice(n_classes, size=n_test_samples, replace=True)

#     rf = RandomForestClassifierCustom(n_classes=n_classes, n_estimators=100, random_state=1)
#     rf.fit(X_train, y_train)
#     y_probs = rf.predict_proba(X_test)