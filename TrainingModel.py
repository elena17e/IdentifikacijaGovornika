from tensorflow import keras
from matplotlib import pyplot as plt


def ResNet_blocks(x, filters, cnn_number = 3, activation = "relu"):
    s = keras.layers.Conv1D(filters, 1, padding = "same")(x)
    
    for i in range(cnn_number - 1):
        x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
        x = keras.layers.Activation(activation)(x)
    
    x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    
    return keras.layers.MaxPool1D(pool_size = 2, strides = 2)(x)

def building(my_shape, number_of_classes):
    inputs = keras.layers.Input(shape = my_shape, name = "input")
    
    x = ResNet_blocks(inputs, 16, 2)
    x = ResNet_blocks(inputs, 32, 2)
    x = ResNet_blocks(inputs, 64, 3)
    x = ResNet_blocks(inputs, 128, 3)
    x = ResNet_blocks(inputs, 128, 3)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    
    outputs = keras.layers.Dense(number_of_classes, activation = "softmax", name = "output")(x)
    
    return keras.models.Model(inputs = inputs, outputs = outputs)

def training(sample_rate, class_names, train_variable, number_of_epochs, test_variable):
    model = building((sample_rate // 2, 1), len(class_names))
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
    model_save_filename = "model.h5"
    earlystopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint(model_save_filename, monitor="val_accuracy", save_best_only=True)

    #treniranje
    histr = model.fit(
        train_variable,
        epochs=number_of_epochs,
        validation_data=test_variable,
        callbacks=[earlystopping, checkpoint],
    )
    print("Model evaluation.")
    plt.title('Loss')
    plt.plot(histr.history['loss'], 'r')
    plt.plot(histr.history['val_loss'], 'b')
    plt.show()
    plt.title('Precision')
    plt.plot(histr.history['precision'], 'r')
    plt.plot(histr.history['val_precision'], 'b')
    plt.show()
    plt.title('Recall')
    plt.plot(histr.history['recall'], 'r')
    plt.plot(histr.history['val_recall'], 'b')
    plt.show()



