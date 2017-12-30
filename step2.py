import json#cPickle as pickle
import cv2
import numpy as np
from sys import stdout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from scipy.misc import imsave
import time
from keras import backend as K



from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets




from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)





def run(train_tensors, train_targets, valid_tensors, valid_targets, test_tensors, test_targets):
    # Keep record of test accuracy. ########################################
    #accHistory = {}



    # Hyper parameter history. #############################################
    hpHistory = []
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':2,
                        'l1_strides':1,
                        'l1_padding':'valid',
                        'l1_poolSize':2,
                        'l1_poolStrides':2,
                      
                        'l2_filters':32,
                        'l2_kSize':4,
                        'l2_strides':1,
                        'l2_padding':'valid',
                        'l2_poolSize':2,
                        'l2_poolStrides':2,
                      
                        'l3_filters':64,
                        'l3_kSize':8,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':2,
                        'l3_poolStrides':2}
                    )
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':4,
                        'l1_strides':1,
                        'l1_padding':'valid',
                        'l1_poolSize':2,
                        'l1_poolStrides':2,
                      
                        'l2_filters':32,
                        'l2_kSize':4,
                        'l2_strides':1,
                        'l2_padding':'valid',
                        'l2_poolSize':2,
                        'l2_poolStrides':2,
                      
                        'l3_filters':64,
                        'l3_kSize':4,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':2,
                        'l3_poolStrides':2}
                    )
    
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':4,
                        'l1_strides':2,
                        'l1_padding':'valid',
                        'l1_poolSize':2,
                        'l1_poolStrides':2,
                      
                        'l2_filters':32,
                        'l2_kSize':4,
                        'l2_strides':2,
                        'l2_padding':'valid',
                        'l2_poolSize':2,
                        'l2_poolStrides':2,
                      
                        'l3_filters':64,
                        'l3_kSize':4,
                        'l3_strides':2,
                        'l3_padding':'valid',
                        'l3_poolSize':2,
                        'l3_poolStrides':2}
                    )
    
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':4,
                        'l1_strides':4,
                        'l1_padding':'valid',
                        'l1_poolSize':2,
                        'l1_poolStrides':2,
                      
                        'l2_filters':32,
                        'l2_kSize':4,
                        'l2_strides':2,
                        'l2_padding':'valid',
                        'l2_poolSize':2,
                        'l2_poolStrides':2,
                      
                        'l3_filters':64,
                        'l3_kSize':4,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':2,
                        'l3_poolStrides':2}
                    )
    
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':8,
                        'l1_strides':1,
                        'l1_padding':'valid',
                        'l1_poolSize':4,
                        'l1_poolStrides':4,
                      
                        'l2_filters':32,
                        'l2_kSize':4,
                        'l2_strides':1,
                        'l2_padding':'valid',
                        'l2_poolSize':4,
                        'l2_poolStrides':4,
                      
                        'l3_filters':64,
                        'l3_kSize':2,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':4,
                        'l3_poolStrides':4}
                    )
    
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':8,
                        'l1_strides':1,
                        'l1_padding':'valid',
                        'l1_poolSize':4,
                        'l1_poolStrides':4,
                      
                        'l2_filters':32,
                        'l2_kSize':8,
                        'l2_strides':1,
                        'l2_padding':'valid',
                        'l2_poolSize':4,
                        'l2_poolStrides':4,
                      
                        'l3_filters':64,
                        'l3_kSize':8,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':4,
                        'l3_poolStrides':4}
                    )
    
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':8,
                        'l1_strides':1,
                        'l1_padding':'valid',
                        'l1_poolSize':4,
                        'l1_poolStrides':4,
                      
                        'l2_filters':32,
                        'l2_kSize':8,
                        'l2_strides':1,
                        'l2_padding':'valid',
                        'l2_poolSize':2,
                        'l2_poolStrides':2,
                      
                        'l3_filters':64,
                        'l3_kSize':8,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':2,
                        'l3_poolStrides':2}
                    )
    
    
    hpHistory.append({  'l1_filters':16,
                        'l1_kSize':4,
                        'l1_strides':1,
                        'l1_padding':'valid',
                        'l1_poolSize':2,
                        'l1_poolStrides':2,
                      
                        'l2_filters':32,
                        'l2_kSize':4,
                        'l2_strides':1,
                        'l2_padding':'valid',
                        'l2_poolSize':2,
                        'l2_poolStrides':2,
                      
                        'l3_filters':64,
                        'l3_kSize':4,
                        'l3_strides':1,
                        'l3_padding':'valid',
                        'l3_poolSize':2,
                        'l3_poolStrides':2,
                      
                        'l4_filters':64,
                        'l4_kSize':4,
                        'l4_strides':1,
                        'l4_padding':'valid',
                        'l4_poolSize':2,
                        'l4_poolStrides':2}
                    )

    
    


    # Loop through the different param settings. ###########################

    for iSetting in range(len(hpHistory)):
        current_setting = hpHistory[iSetting]
        print('Testing setting {n:g} ***************************************************************************'.format(n = iSetting))
        startTime = time.time()
        print('Setting up model.')
        # Build the CNN. #######################################################
        model = Sequential()

        # First convolutional layer.
        model.add( Conv2D(  filters = hpHistory[iSetting]['l1_filters'],
                            kernel_size = hpHistory[iSetting]['l1_kSize'],
                            strides = hpHistory[iSetting]['l1_strides'],
                            padding = hpHistory[iSetting]['l1_padding'],
                            activation = 'relu',
                            input_shape=train_tensors[0].shape,
                            name = 'conv_1'
                         )
                 )
        model.add( MaxPooling2D( pool_size = hpHistory[iSetting]['l1_poolSize'],
                                 strides = hpHistory[iSetting]['l1_poolStrides'],
                                 padding = hpHistory[iSetting]['l1_padding'],
                                 name = 'pool_1'
                               )
                 )

        # Second convolutional layer.
        if 'l2_kSize' in hpHistory[iSetting].keys():
            model.add( Conv2D(  filters = hpHistory[iSetting]['l2_filters'],
                                kernel_size = hpHistory[iSetting]['l2_kSize'],
                                strides = hpHistory[iSetting]['l2_strides'],
                                padding = hpHistory[iSetting]['l2_padding'],
                                activation = 'relu',
                                name = 'conv_2' ))
            model.add( MaxPooling2D( pool_size = hpHistory[iSetting]['l2_poolSize'],
                                     strides = hpHistory[iSetting]['l2_poolStrides'],
                                     padding = hpHistory[iSetting]['l2_padding'],
                                     name = 'pool_2' ))
        # Third convolutional layer.
        if 'l3_kSize' in hpHistory[iSetting].keys():
            model.add( Conv2D(  filters = hpHistory[iSetting]['l3_filters'],
                                kernel_size = hpHistory[iSetting]['l3_kSize'],
                                strides = hpHistory[iSetting]['l3_strides'],
                                padding = hpHistory[iSetting]['l3_padding'],
                                activation = 'relu',
                                name = 'conv_3' ))
            model.add( MaxPooling2D( pool_size = hpHistory[iSetting]['l3_poolSize'],
                                     strides = hpHistory[iSetting]['l3_poolStrides'],
                                     padding = hpHistory[iSetting]['l3_padding'],
                                     name = 'pool_3' ))
        
        # Fourth convolutional layer.
        if 'l4_kSize' in hpHistory[iSetting].keys():
            model.add( Conv2D(  filters = hpHistory[iSetting]['l4_filters'],
                                kernel_size = hpHistory[iSetting]['l4_kSize'],
                                strides = hpHistory[iSetting]['l4_strides'],
                                padding = hpHistory[iSetting]['l4_padding'],
                                activation = 'relu',
                                name = 'conv_4' ))
            model.add( MaxPooling2D( pool_size = hpHistory[iSetting]['l4_poolSize'],
                                     strides = hpHistory[iSetting]['l4_poolStrides'],
                                     padding = hpHistory[iSetting]['l4_padding'],
                                     name = 'pool_4' ))
            
        # Add global pooling layer.
        model.add( GlobalAveragePooling2D() )
            
        # Add classification layer.
        model.add( Dense(133, activation='softmax') )
        model.summary()

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


        
        # Train the model. #####################################################
        print('')
        print('Training model.')
        epochs = 5

        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)

        model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

        train_time = time.time() - startTime
        
        # Load the best weights.
        model.load_weights('saved_models/weights.best.from_scratch.hdf5')



        # Visualize the weights. ###############################################
        print('')
        print('Creating weight images.')
        # dimensions of the generated pictures for each filter.
        img_width = train_tensors[0].shape[0]
        img_height = train_tensors[0].shape[1]

        # util function to convert a tensor into a valid image
        def deprocess_image(x):
            # normalize tensor: center on 0., ensure std is 0.1
            x -= x.mean()
            x /= (x.std() + K.epsilon())
            x *= 0.1

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            # convert to RGB array
            x *= 255
            if K.image_data_format() == 'channels_first':
                x = x.transpose((1, 2, 0))
            x = np.clip(x, 0, 255).astype('uint8')
            return x

        # this is the placeholder for the input images
        input_img = model.input

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])

        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())



        # The name of the layer we want to visualize.
        layer_names =  []
        for name in layer_dict:
            if 'conv' in name:
                layer_names.append(name)
        #layer_name = 'conv_1'

        # Create weight images for each convolutional layer.
        for layer_name in layer_names:
            print('   Creating weight image for layer {n:s}'.format(n = layer_name))
            n_filters = layer_dict[layer_name].filters
            kept_filters = []
            for filter_index in range(n_filters):
                print('      Processing filter %d' % filter_index)
                start_time = time.time()

                # we build a loss function that maximizes the activation
                # of the nth filter of the layer considered
                layer_output = layer_dict[layer_name].output
                if K.image_data_format() == 'channels_first':
                    loss = K.mean(layer_output[:, filter_index, :, :])
                else:
                    loss = K.mean(layer_output[:, :, :, filter_index])

                # we compute the gradient of the input picture wrt this loss
                grads = K.gradients(loss, input_img)[0]

                # normalization trick: we normalize the gradient
                grads = normalize(grads)

                # this function returns the loss and grads given the input picture
                iterate = K.function([input_img], [loss, grads])

                # step size for gradient ascent
                step = 1.

                # we start from a gray image with some random noise
                if K.image_data_format() == 'channels_first':
                    input_img_data = np.random.random((1, 3, img_width, img_height))
                else:
                    input_img_data = np.random.random((1, img_width, img_height, 3))
                input_img_data = (input_img_data - 0.5) * 20 + 128

                # we run gradient ascent for 20 steps
                for i in range(30):
                    loss_value, grads_value = iterate([input_img_data])
                    input_img_data += grads_value * step

                    #print('Current loss value:', loss_value)
                    stdout.write('{r:s}         Current loss value: {n:2.2f}'.format(r = '\r', n = loss_value))
                    stdout.flush()
                    if loss_value <= 0.:
                        # some filters get stuck to 0, we can skip them
                        break
                print('')

                # Decode the resulting input image.
                img = deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
                end_time = time.time()
                print('      Filter %d processed in %ds' % (filter_index, end_time - start_time))


            # Create the image and save it.
            n = 8
            if n_filters <=36:
                n = 6
                if n_filters <= 25:
                    n = 5
                    if n_filters <= 16:
                        n = 4
                        if n_filters <= 9:
                            n = 3
                            if n_filters <=4:
                                n = 2

            # The filters that have the highest loss are assumed to be better-looking. Sort by loss.
            kept_filters.sort(key=lambda x: x[1], reverse=True)

            # Build a black picture with enough space for all filter images.
            # Keep 5px margin between pictures.
            margin = 5
            width = n * img_width + (n - 1) * margin
            height = n * img_height + (n - 1) * margin
            stitched_filters = np.zeros((width, height, 3))

            # fill the picture with our saved filters
            for i in range(n):
                for j in range(n):
                    try:
                        img, loss = kept_filters[i * n + j]
                        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
                    except IndexError:
                        pass

            # Save the result to disk
            print('   Saving image.')
            cv2.imwrite('weightImages/hp{n:g}_{l:s}.png'.format(n = iSetting, l = layer_name), stitched_filters)



        # Test the CNN. ######################################################
        # get index of predicted dog breed for each image in test set
        dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

        # Report test accuracy
        test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
        print('')
        print('Test accuracy: %.4f%%' % test_accuracy)

        hpHistory[iSetting]['accuracy'] = test_accuracy
        hpHistory[iSetting]['time'] = train_time
        hpHistory[iSetting]['i'] = iSetting
        
        # Save the results.
        with open('results', 'w') as file:
             file.write(json.dumps(hpHistory))
                
        print('Done in {n:g} seconds.'.format(n = time.time() - startTime))
        print('')
        print('')
        


if __name__ == "__main__":
        
    print('Loading data.')
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]



    print('Preparing tensors.')
    from PIL import ImageFile                            
    ImageFile.LOAD_TRUNCATED_IMAGES = True                 

    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255


    print('Running.')
    run(train_tensors, train_targets, valid_tensors, valid_targets, test_tensors, test_targets)