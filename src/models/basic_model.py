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
			layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
			layers.MaxPool2D((2, 2)),
			layers.Flatten(),
			layers.Softmax(axis=categories_count),
		])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
