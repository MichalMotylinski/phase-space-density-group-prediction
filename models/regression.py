from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt


def random_forest(data, output):
    x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=0.2)
    regr = RandomForestRegressor(max_depth=30, n_jobs=-1)
    regr.fit(x_train, y_train)

    print(regr.predict(x_test))
    print(y_test)
    print(regr.score(x_test, y_test))
    print(y_test[:20])
    print(regr.predict(x_test)[:20])


class RadialVelocityRegression:
    def get_initializer(self):
        return tf.keras.initializers.GlorotUniform()

    def velocity_layers(self, inputs):
        setup = Dense(30, activation="tanh", kernel_initializer=self.get_initializer(), input_dim=5)(inputs)
        setup = Dense(units=30, activation='tanh', kernel_initializer=self.get_initializer())(setup)
        setup = Dense(units=30, activation='tanh', kernel_initializer=self.get_initializer())(setup)
        setup = Dense(units=30, activation='tanh', kernel_initializer=self.get_initializer())(setup)
        setup = Dense(units=1, kernel_initializer=self.get_initializer(), activation="linear")(setup)

        return setup

    def uncertainty_layers(self, inputs):
        setup = Dense(30, activation="relu", kernel_initializer=self.get_initializer(), input_dim=5)(inputs)
        setup = Dense(units=30, activation='relu', kernel_initializer=self.get_initializer())(setup)
        setup = Dense(units=30, activation='relu', kernel_initializer=self.get_initializer())(setup)
        setup = Dense(units=30, activation='relu', kernel_initializer=self.get_initializer())(setup)
        setup = Dense(units=1, kernel_initializer=self.get_initializer(), activation="linear")(setup)

        return setup

    def assemble_model(self, input_shape):
        inputs = Input(shape=input_shape)

        velocity_branch = self.velocity_layers(inputs)
        uncertainty_branch = self.uncertainty_layers(inputs)

        model = Model(inputs=inputs, outputs=[velocity_branch, uncertainty_branch], name="rv_net")

        lr = 1e-3
        epochs = 100
        optimizer = Adam(lr=lr)

        model.compile(optimizer=optimizer,
                      loss='mean_squared_error'
                      )

        return model

    def train_model(self, x_train, y_train, x_val, y_val, model):
        batch_size = 64
        valid_batch_size = 64
        epochs = 10

        history = model.fit(x_train,
                            y_train,
                            validation_data=(x_val, y_val),
                            batch_size=batch_size,
                            validation_batch_size=valid_batch_size,
                            epochs=epochs)

        return history, model

    def train_test_split(self, data, output):
        x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=0.2)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_val = sc.fit_transform(x_val)
        x_test = sc.transform(x_test)

        return x_train, y_train, x_val, y_val, x_test, y_test


def ann(data, output):
    x_train, x_test, y_train, y_test = train_test_split(data, output, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_val = sc.fit_transform(x_val)
    x_test = sc.transform(x_test)

    initializer = tf.keras.initializers.GlorotUniform()

    model = Sequential()
    model.add(Dense(30, activation='tanh', kernel_initializer=initializer, input_dim=5))
    model.add(Dense(units=30, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(units=30, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(units=30, activation='tanh', kernel_initializer=initializer))
    model.add(Dense(units=1, kernel_initializer=initializer, activation="linear"))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=10)

    y_pred = model.predict(x_test)

    print(y_test[:20])
    print(y_pred[:20])


