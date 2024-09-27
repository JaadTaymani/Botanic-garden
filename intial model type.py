def prepare_image(image_path):
    '''
    turns an image into a tensor
    '''
    # read an image
    image = tf.io.read_file(image_path)
    # turn an image to numerical version
    image = tf.image.decode_jpeg(image, channels=3)
    # convert colours from 0-255 to 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize
    image = tf.image.resize(image, size=[img_size, img_size])
    
    return image


def create_batches(X, y=None, batch_size=batch_size, test_data=False):
    '''
    split a dataset to batches
    '''
    if test_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_label_image).batch(batch_size)
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_label_image).batch(batch_size)
    
    return data_batch


# batches creation for all datasets

# train data
train_data = create_batches(X_train, y_train)

# test data
test_data = create_batches(X_test, y_test, test_data=True)

# validation data
val_data = create_batches(X_val, y_val, test_data=True)


# model preparation
model_1 = tf.keras.Sequential([
    tf.keras.layers.Input(input_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(output_shape)
])

model_1.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_1.summary()
model_1.fit(train_data, 
            epochs=5,
            validation_data=val_data, 
            validation_freq=1,
           )