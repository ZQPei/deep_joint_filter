import cv2
import os
import glob
import skimage
import numpy as np
 
def add_gaussian_noise(image_in, noise_sigma=25):
    temp_image = np.float64(np.copy(image_in))
 
    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
 
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image


if __name__ == "__main__":
    img_path = os.getcwd()+'/data/depart/label/1/0.png'
    img = cv2.imread(img_path,0)
    noise_sigma = 25
    noise_img = add_gaussian_noise(img,noise_sigma=noise_sigma)
    cv2.imwrite('noise_{}.png'.format(noise_sigma),noise_img)
