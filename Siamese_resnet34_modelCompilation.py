import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define Residual Block function
def residual_block(x, filters, blocks, stride):
    shortcut = x
    for i in range(blocks):
        if i == 0:
            shortcut = Conv2D(filters, (1, 1), strides=stride)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Conv2D(filters, (3, 3), padding='same', strides=stride if i == 0 else 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        x = Add()([x, shortcut])
        x = tf.nn.relu(x)  # Apply ReLU activation after addition
        shortcut = x
    
    return x

# Build ResNet-34 architecture
def build_resnet_34(input_shape):
    inputs = Input(shape=input_shape)
    
    # Stage 1
    x = Conv2D(64, (7, 7), padding='same', strides=2, activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    
    # Stage 2
    x = residual_block(x, filters=64, blocks=3, stride=1)
    
    # Stage 3
    x = residual_block(x, filters=128, blocks=4, stride=2)
    
    # Stage 4
    x = residual_block(x, filters=256, blocks=6, stride=2)
    
    # Stage 5
    x = residual_block(x, filters=512, blocks=3, stride=2)
    
    # Global average pooling and dense layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(128, activation='relu')(x)
    
    model = Model(inputs, outputs)
    return model

# Contrastive Loss function
def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# Build ResNet-34 base network
input_shape = (150, 150, 3)
base_network = build_resnet_34(input_shape)

# Define Siamese model inputs
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# Process inputs through the base network
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Calculate L1 distance between processed inputs
distance = tf.abs(processed_a - processed_b)

# Predict distances with a Dense layer (linear activation)
outputs = Dense(1, activation='linear')(distance)

# Define Siamese model
siamese_model = Model([input_a, input_b], outputs)

# Compile Siamese model with contrastive loss
siamese_model.compile(optimizer=Adam(), loss=contrastive_loss, metrics=['accuracy'])

# Print total number of layers in the ResNet-34 base network
print("Total number of layers in ResNet-34 base network:", len(base_network.layers))

# Print model summary
print("\nSiamese Model Summary:")
siamese_model.summary()

# Save compiled model
siamese_model.save('siamese_resnet34_contrastive_model.h5')

print("\nSiamese model with ResNet-34 base and contrastive loss compiled and saved successfully.")
