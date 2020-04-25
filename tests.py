import tensorflow

from transformer import positional_encoding

def test_encoding_relative_positions(embedding_size=15, d_model=32, number_samples=1000, number_positions=30):

    def create_dataset(number_samples,number_positions):
        X = []
        y = []
        for _ in range(number_samples):
            positions = np.random.randint(number_positions, size=2)
            embedding_max = positional_encoding(positions.max(), d_model)
            embedding_min = positional_encoding(positions.min(), d_model)
            X.append(np.concatenate([embedding_max, embedding_min]))
            y.append(positions.max() - positions.min())
        X = np.array(X)
        y = np.array(y)
        X_tr, X_val = X[:int(0.75 * len(X))], X[int(0.75 * len(X)):]
        y_tr, y_val = y[:int(0.75 * len(y))], y[int(0.75 * len(y)):]
        return X_tr, X_val, y_tr, y_val

    def create_model(embedding_size):
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Input(shape=(2 * embedding_size)),
            tensorflow.keras.layers.Dense(16, activation="relu"),
            tensorflow.keras.layers.Dense(8, activation="relu"),
            tensorflow.keras.layers.Dense(1, activation="relu")
        ])
        model.compile(
                    optimizer=tensorflow.keras.optimizers.Adam(lr=0.001),
                    loss='mse',
                )
        return model

    X_tr, X_val, y_tr, y_val = create_dataset(number_samples,number_positions)
    model = create_model(embedding_size)
    history_callback = model.fit(X_tr, y_tr, epochs=100, validation_data=(X_val, y_val))
    assert history_callback.history["val_loss"][-1] < 0.05
