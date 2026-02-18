import keras
from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = keras.Sequential([
			layers.Input(shape=input_shape),
			layers.Rescaling(scale=1. / 255),
			layers.Conv2D(16, (3, 3), activation='relu'),
			layers.MaxPool2D((2, 2)),
			layers.Conv2D(32, (3, 3), activation='relu'),
			layers.MaxPool2D((2, 2)),
			layers.Dropout(rate=0.2),
			layers.Conv2D(32, (6, 6), activation='relu'),
			layers.MaxPool2D((2, 2)),
			layers.Flatten(),
			layers.Dense(categories_count, activation='softmax'),
		])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
