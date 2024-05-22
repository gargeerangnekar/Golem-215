import pandas as pd
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img
import numpy as np
# Load CSV file
csv_path = "train.csv"
df = pd.read_csv(csv_path)

# Set image size and batch size
img_size = (128, 128)
batch_size = 32

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory="D:/sd/DataSet/Train_Images",
    x_col="Image_File",
    y_col="Class",       
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory="D:/sd/DataSet/Train_Images/",
    x_col="Image_File",
    y_col="Class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

epochs = 10
#model.fit(train_generator, epochs=epochs, validation_data=test_generator)
img_size = (128, 128)
batch_size = 32

#print(new_df)
model_path = "rock_classifier.h5"
model = load_model(model_path)

new_csv_path = 'test.csv'
new_df = pd.read_csv(new_csv_path)

new_image_paths = ['D:/sd/DataSet/Test_Images/' + img_name.strip() for img_name in new_df['Image_File']]

predictions = []

for image_path in new_image_paths:
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 

    prediction = model.predict(img_array)[0, 0]
    predictions.append(prediction)

new_df['prediction'] = predictions
threshold = 0.5
new_df['Class'] = np.where(new_df['prediction'] > threshold, 'large', 'small')
print(new_df['Image_File'], new_df['Class'])
output_csv_path = 'predicted_test.csv'
test_df.to_csv(output_csv_path, index=False)