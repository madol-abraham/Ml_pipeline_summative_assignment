from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

def define_model(early_stopping: bool, dropout: float, learning_rate: float, num_features: int, optimization: str = "adam", regularization: float = 0.01, reg_strength: float = 0.01):
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(num_features,)))

    # Hidden Layers
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(reg_strength)))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(regularization)))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))

    # Output Layer (Binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Optimizer selection
    optimizers_dict = {
        "adam": Adam(learning_rate=learning_rate),
        "sgd": SGD(learning_rate=learning_rate),
        "rmsprop": RMSprop(learning_rate=learning_rate)
    }
    optimizer = optimizers_dict.get(optimization.lower(), Adam(learning_rate=learning_rate))

    # Compile model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Early Stopping Callback (if enabled)
    callback_list = [EarlyStopping(monitor='val_loss', patience=5)] if early_stopping else []

    return model, callback_list

# Now your model should run without errors
num_features = X_train.shape[1]  # Ensure this is defined

model, callbacks = define_model(early_stopping=True,
    dropout=0.5,
    learning_rate=0.001,
    num_features=num_features,
    optimization="adam",
    reg_strength=0.01
                                ) # Now passing a float value
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=callbacks)

# save the model
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')  # Load the saved model
