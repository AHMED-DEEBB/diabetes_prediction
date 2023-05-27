# Import data
import numpy as np

#load the data
data = np.loadtxt("E:\Ahmed's Projects\diabetes prediction\diabetes records.csv", delimiter=',')
print(data)

input_data = data[:,0:8]
output_data = data[:, 8]
print(output_data[0])

# Build Machine Learning model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Set the configuration
model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(input_data, output_data, epochs=150, batch_size=10)

# Calculating the accuracy
accuracy = model.evaluate(input_data, output_data)
print("Accuracy: %.2f%%" % (accuracy[1] * 100))

# Make predictions
predictions = (model.predict(input_data) > 0.5).astype(int)

# Compare the predicted data with the actual data
def Display(num):
    if num == 0:
        return "no disease"
    else:
        return "disease"

for i in range(10):
    print("%s => %s (expected %s)" % (input_data[i].tolist(), Display(predictions[i]), Display(output_data[i])))