import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

def generate_aug_images(source_path, output_path, set, label):
    # Set the augmentation conditions
    image_aug_gen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=10,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.7,
                                        channel_shift_range=10,
                                        zoom_range=[0.9, 2.2],
                                        horizontal_flip=True,
                                        fill_mode='nearest')
    
    image_files = os.listdir(source_path + '\\' + set + '\\' + label)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(output_path + ' is created.')

    if not os.path.exists(output_path + '\\' + set):
        os.makedirs(output_path + '\\' + set)
        print(output_path + '\\' + set + ' is created.')
    
    if not os.path.exists(output_path + '\\' + set + '\\' + label):
        os.makedirs(output_path + '\\' + set + '\\' + label)
        print(output_path + '\\' + set + '\\' + label + ' is created.')

    # Generate and save augemented images
    for image_file in image_files:
        image = load_img(source_path + '\\' + set + '\\' + label + '\\' + image_file)
        image_array = img_to_array(image)
        image_array = image_array.reshape((1,) + image_array.shape)

        i = 0

        print(set + '/' + label + '/' + image_file + ' is processing...')

        for batch in image_aug_gen.flow(image_array, batch_size=100,
                                   save_to_dir=output_path + '\\' + set + '\\' + label,
                                   save_prefix=os.path.splitext(image_file)[0],
                                   save_format='jpg'):
            i += 1
            if i > 49:
                break

if __name__ == '__main__':
    dataset_path = r'..\dataset\FDM_Process_Image_Dataset_v1'
    aug_dataset_path = r'..\dataset\FDM_Process_Image_Dataset_v1_aug'
    sets = ['train', 'valid']
    labels = ['success', 'failure']

    for set in sets:
        for label in labels:
            generate_aug_images(dataset_path, aug_dataset_path, set, label)
    
    ### TO DO: After generating augmented images of each label, you should copy the test set folder in the same location. ###
