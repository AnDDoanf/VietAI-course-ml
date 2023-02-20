import math


class LinearRegressionFC:
    def __init__(self):
        self.weights = None
        self.train_loss = []
        self.y_predict = []
        self.val_loss = []

    def __hypothesis(self, W, X_r):
        '''
        Parameters
        ----------
        X_r: stands for a row in dataset
        '''
        y_hat = 0
        for idx, feature_value in enumerate(X_r):
            y_hat += W[idx] * feature_value
        return y_hat

    def __loss_func(self, W, X, y):
        l = 0
        for idx, row in enumerate(X):
            y_hat = self.__hypothesis(W, row)
            l += (0.5 * pow((y_hat - y[idx]), 2) / len(X))
        return l

    def fit(self, X, y, lr, epochs, X_val, y_val):
        for row in X:
            row.insert(0, 1)
        init_w = [1] * len(X[0])

        for _ in range(epochs):
            for row_idx, row in enumerate(X):
                for idx, w in enumerate(init_w):
                    if idx == 0:
                        init_w[idx] = w - lr * (1/len(X)) * (self.__hypothesis(init_w, row) - y[row_idx])
                    else:
                        init_w[idx] = w - lr * (1/len(X)) * (self.__hypothesis(init_w, row) - y[row_idx]) * row[idx]
            # print(f'Epoch {i}, loss: ', self.__loss_func(init_w, X, y))
            self.train_loss.append(self.__loss_func(init_w, X, y))
            self.val_loss.append(self.__loss_func(init_w, X_val, y_val))
        self.weights = init_w

    def predict(self, data_point):
        prediction = 0
        for i, feature in enumerate(data_point):
            prediction += self.weights[i] * feature
        return prediction

    def evaluate(self, X_val, y_val):
        rmse = 0
        for i, row in enumerate(X_val):
            y_val_predicted = self.predict(row)
            self.y_predict.append(y_val_predicted)
            rmse += pow((y_val[i] - y_val_predicted), 2)
        return (math.sqrt(rmse/len(X_val)) / len(y_val))
