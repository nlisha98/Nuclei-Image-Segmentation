#%%
#1. Import packages
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping, ModelCheckpoint
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os, glob, cv2, datetime
import numpy as np

# %%
#2. Load Images
FILE_TRAIN_PATH = os.path.join(os.getcwd(),'dataset','train')
FILE_TEST_PATH = os.path.join(os.getcwd(), 'dataset', 'test')

TRAIN_IMAGE_PATH = os.path.join(FILE_TRAIN_PATH,'inputs')
TRAIN_MASK_PATH=os.path.join(FILE_TRAIN_PATH,'masks')
TEST_IMAGE_PATH = os.path.join(FILE_TEST_PATH, 'inputs')
TEST_MASK_PATH = os.path.join(FILE_TEST_PATH, 'masks')

train_images = []
for img in os.listdir(TRAIN_IMAGE_PATH):
    #Get full path of image file
    FULL_PATH = os.path.join(TRAIN_IMAGE_PATH,img)
    #Read image file based on the full path
    img_np = cv2.imread(FULL_PATH)
    #Convert image from BGR to RGB
    img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    #Resize image into 128x128
    img_np = cv2.resize(img_np,(128,128))
    #Place image into empty list
    train_images.append(img_np)

test_images = []
for img in os.listdir(TEST_IMAGE_PATH):
    #Get full path of image file
    FULL_PATH = os.path.join(TEST_IMAGE_PATH,img)
    #Read image file based on the full path
    img_np = cv2.imread(FULL_PATH)
    #Convert image from BGR to RGB
    img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    #Resize image into 128x128
    img_np = cv2.resize(img_np,(128,128))
    #Place image into empty list
    test_images.append(img_np)

# %%
#3. Load masks
train_masks = []
for mask in os.listdir(TRAIN_MASK_PATH):
    #Get full path of the mask file
    FULL_PATH=os.path.join(TRAIN_MASK_PATH,mask)
    #Read the mask file as a grayscale image
    mask_np=cv2.imread(FULL_PATH,cv2.IMREAD_GRAYSCALE)
    #Resize the image into 128x128
    mask_np=cv2.resize(mask_np,(128,128))
    #Place the mask into the empty list
    train_masks.append(mask_np)

test_masks = []
for mask in os.listdir(TEST_MASK_PATH):
    #Get full path of the mask file
    FULL_PATH=os.path.join(TEST_MASK_PATH,mask)
    #Read the mask file as a grayscale image
    mask_np=cv2.imread(FULL_PATH,cv2.IMREAD_GRAYSCALE)
    #Resize the image into 128x128
    mask_np=cv2.resize(mask_np,(128,128))
    #Place the mask into the empty list
    test_masks.append(mask_np)

# %%
#4. Convert list into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
#4.1. Check some examples
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_images_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(train_masks_np[i])
    plt.axis('off')
    
plt.show()

# %%
#5. Data Preprocessing
#5.1. Expand the mask dimension to include the channel axis
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
#5.2 Convert mask value into 0 and 1
train_converted_masks_np = np.round(train_masks_np_exp/255).astype(np.int64)
test_converted_masks_np = np.round(test_masks_np_exp/255).astype(np.int64)
#5.3 Normalize the images pixel value
train_normalized_images_np = train_images_np/255.0
test_normalized_images_np = test_images_np/255.0

# %%
#6. Perform train test split
SEED = 12345
X_train, X_val, y_train, y_val = train_test_split(train_normalized_images_np,train_converted_masks_np, test_size=0.2, random_state=SEED)

# %%
#7. Convert numpy array into TensorFlow tensors
train_input_tensor=tf.data.Dataset.from_tensor_slices(X_train)
train_masks_tensor=tf.data.Dataset.from_tensor_slices(y_train)
val_input_tensor=tf.data.Dataset.from_tensor_slices(X_val)
val_mask_tensor=tf.data.Dataset.from_tensor_slices(y_val)
test_input_tensor = tf.data.Dataset.from_tensor_slices(test_normalized_images_np)
test_masks_tensor = tf.data.Dataset.from_tensor_slices(test_converted_masks_np)

# %%
#8. Combine features and labels together to form a zip dataset
train=tf.data.Dataset.zip((train_input_tensor,train_masks_tensor))
val=tf.data.Dataset.zip((val_input_tensor,val_mask_tensor))
test=tf.data.Dataset.zip((test_input_tensor,test_masks_tensor))

#%%
#Create subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels

# %%
#9. Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

val_batches=val.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


#%%
#10. Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15,15))
    title=['Input Image','True Mask','Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.show()

for images, masks in train_batches.take(2):
    sample_image, sample_mask=images[0],masks[0]
    display([sample_image, sample_mask])

# %%
#11. Model Development
#11.1. Use a pretrained model as the feature extractor
base_model=tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#11.2 Use these activation layers as the output from the feature extractor (some of these outputs will be use to perform concatenation with the upsampling path)
layer_names=[
    'block_1_expand_relu', #64x64
    'block_3_expand_relu', #32x32
    'block_6_expand_relu', #16x16
    'block_13_expand_relu', #8x8
    'block_16_project',     #4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# %%
#11.3 Instantiate the feature extractor
down_stack=tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# %%
#11.4 Define the upsampling path
up_stack=[
    pix2pix.upsample(512,3), #4x4 --> 8x8
    pix2pix.upsample(256,3), #8x8 --> 16x16
    pix2pix.upsample(128,3), #16x16 --> 32x32
    pix2pix.upsample(64,3)  #32x32 --> 64x64
]

# %%
#11.5. Use functional API to construct the entire UNet
def unet(output_channels:int):
    inputs=tf.keras.layers.Input(shape=[128,128,3])
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips=reversed(skips[:-1])

    #Build the upsampling path and establishing the skip connections (concatenation)
    for up, skip in zip(up_stack,skips):
        x=up(x)
        concat=tf.keras.layers.Concatenate()
        x=concat([x,skip])

    #Use a transpose convolution layer to perform one last upsampling (Output Layer)
    last=tf.keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    outputs = last(x)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)

    return model 

# %%
#11.6. Create the model using the above function
OUTPUT_CLASSES=2
model = unet(OUTPUT_CLASSES)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
#%%
#12. Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

#%%
# %%
#13.Create functions to show predictions
def create_mask(pred_mask):
    pred_mask=tf.argmax(pred_mask,axis=-1)
    pred_mask=pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask=model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])

    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

show_predictions()

#%%
#14. Create a callback function to make use of the show_predictions() function
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample prediction after epoch {}\n'.format(epoch+1))

#Callbacks - Early Stopping and TensorBoard
LOGS_PATH=os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback= TensorBoard(log_dir=LOGS_PATH)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(os.path.join(os.getcwd(), 'checkpoint'), monitor='val_acc', save_best_only=True)

# %%
#15. Model training
EPOCHS=15
VAL_SUBSPLITS=5
VALIDATION_STEPS=len(val)//BATCH_SIZE//VAL_SUBSPLITS

model_history=model.fit(train_batches,validation_data=val_batches,validation_steps=VALIDATION_STEPS,epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,callbacks=[DisplayCallback(),tensorboard_callback, early_stop_callback, model_checkpoint])

# %%
#16. Model Deployment 
show_predictions(test_batches,num=5)

#%%
#Evaluate model with test data
print(model.evaluate(test_batches))
#%%
#17. Model Saving
#Save deep learning model
model.save('saved_models.h5')
# %%
