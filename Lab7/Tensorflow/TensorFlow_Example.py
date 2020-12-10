'''
    Lab 7: Using TensorFlow for Image Classification
    Projects: Basic Classification: Classify Images of Clothing
'''

#Import TensorFlow
import tensorflow as tf

#Import helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Function to plot the value in the image
def plot_value_array(i, predictions_array, true_label):
    '''
        Purpose: Plot the value array of label of the image
        Input: i - int64 -the index of the image
                prediction_array - np.array - the array of predictedlabel
                true_label - np.array - the array of true label
        Output: The expected plot figure
    '''
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#Function to plot the predicted image
def plot_image(i, predictions_array, true_label, img,class_names):
    '''
        Purpose: Plot the predicted image
        Input: i - int64 -the index of the image
                prediction_array - np.array - the array of predictedlabel
                true_label - np.array - the array of true label
                img - np.array - the input image
                class_names - list - the names of the class
        Output: The expected plot figure
    '''
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
    100*np.max(predictions_array),
    class_names[true_label]),
    color=color)

def main():
    #Import the Fashion_mnist dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    '''
        This dataset contains 70,000 grayscale images in 10 categories. The images show
        individual articles of clothing at low resolution (28 by 28 pixels)
    '''
    print('='*80)

    #Load dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print('='*80)

    #Print the shape of the train_images and test_images
    print('The shape of the train image: ',train_images.shape)
    print('The shape of the test image: ',test_images.shape)
    print('='*80)
    '''
        As we can see, the shape of train image is (60000,28,28). So The train image 
        contains 60000 grayscale images with 28 by 28 pixels. 

        The shape of test image is (10000, 28,28). So the test image contains 10000
        grayscale images with 28 by 28 pixels

        The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255.

        The labels are an array of integers, ranging from 0 to 9

        Each image is mapped to a single label. Since the class names are not included
        with the dataset, store them here to use later when plotting the images
    '''

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print('='*80)

    #We'll inspect the first image in the training set.
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.savefig('tensorflow_first_training.jpg')
    print('='*80)
    #The pixels values fall in the range of 0 to 255.

    #We need to scale these values to a range of 0 to 1 before feeding them to the 
    #neural network model

    #Scale the train_images
    train_images = train_images / 255.0
    #We do the same thing with test_images
    test_images = test_images / 255.0
    print('='*80)

    #We'll inspect the first 25 images in the train_images and display them with the
    #class label.
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.savefig('tensorflow_25_train.jpg')
    print('='*80)
    #We can see that, all the images are in the correct format. We can now build and 
    #train the network

    ##Build the Model
    '''
        We use neural network to configure the layers and then compiling models

        The basic building block of a neural network is the layer. 
        
        Layers extract representations from the data fed into them.

        Most of deep learning consists of chaining together simple layers. Most layers,
        such as tf.keras.layers.Dense, have parameters that are learned during training.
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    print('='*80)
    '''
        The first layer in this network, tf.keras.layers.Flatten, transform the format
        of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional
        array (of 28 * 28 = 784 pixels).

        After the pixels are flattened, the network consists of a sequence of two 
        tf.keras.layers.Dense layers. These are densely connected, or fully connected,
        neural layers.
    '''

    #Now we compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    print('='*80)
    '''
        Parameters:
        Loss function: measuring how accurate the model is during training.
        Optimizer: how the model is updated based on the data it sees and its loss function.
        Metrics: monitoring the training and testing steps. 
    '''

    ##Train the model
    '''
        Training the neural network model requires the following steps:

        Feed the training data to the model.

        The model learns to associate images and labels.

        Make predictions about a test set—in this example, the test_images array.

        Verify that the predictions match the labels from the test_labels array.
    '''
    #Using model.fit to fit the model to the training data
    model.fit(train_images, train_labels, epochs=10)
    print('='*80)

    #Evaluate the accuracy on test_images
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('='*80)

    #Print the test accuracy
    print('\nTest accuracy of the model:', test_acc)
    print('='*80)
    '''
        As we can see, the accuracy of this model is 0.8778
    '''

    #Now we have the model, we'll make some predictions about some images from the model
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print('='*80)

    #Let's look at the first predicted image
    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images,class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.savefig('tensorflow_first_predict.jpg')
    print('='*80)

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images,class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.savefig('tensorflow_15_predict.jpg')
    print('='*80)

    #Now we can use the model to make prediction of a single image
    #Grab an image from the test dataset.
    img = test_images[1]
    print('='*80)

    #Add the image to a batch where it's the only member.
    img = (np.expand_dims(img,0))
    print('='*80)

    #Predict the correct label for this image
    predictions_single = probability_model.predict(img)
    print('='*80)

    plt.figure(figsize=(10,10))
    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.savefig('tensorflow_correct_label.jpg')
    print('='*80)
    '''
        We can see that the model predicts a label as expected
    '''

if __name__=='__main__':
    main()