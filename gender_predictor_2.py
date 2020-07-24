import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def get_train_test_sets(data_file):
    data = np.genfromtxt(data_file,
                         delimiter=',',
                         encoding='utf-8',
                         dtype=str)
    set_x = data[:, :-1]
    set_y = data[:, -1]
    encoder = OrdinalEncoder()
    set_x = encoder.fit_transform(set_x)
    return train_test_split(set_x, set_y, test_size=0.3)


def main():
    train_x, test_x, train_y, test_y = get_train_test_sets(
        'transformed_data.csv')
    classifier = RandomForestClassifier(n_estimators=50)
    classifier.fit(train_x, train_y)
    score = classifier.score(test_x, test_y)
    print(score)


if __name__ == '__main__':
    main()
