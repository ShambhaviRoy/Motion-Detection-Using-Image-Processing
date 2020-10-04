   
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:18:12 2019

@author: admin
"""
import numpy as np
# Homomorphic filter class
class HomomorphicFilter:
  
    def __init__(self, a = 0.5, b = 1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = 1/(1+(Duv/filter_params[0]**2)**filter_params[1])
        return (1 - H)
   
    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b*H)*I
        return I_filtered

    def filter(self, I, filter_params, H = None):
        #  Validating image
        if len(I.shape) is not 2:
            raise Exception('Improper image')

        # Take the image to log domain and then to frequency domain 
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)
        H = self.__butterworth_filter(I_shape = I_fft.shape, filter_params = filter_params)
        
        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I = I_fft, H = H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt))-1
        return np.uint8(I)
# End of class HomomorphicFilter

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from skimage.filters import threshold_otsu
    
    # Main code
    original_1 = cv2.imread('2019-01-10_09_20_03.jpeg',1)
    original_2 = cv2.imread('2019-01-10_09_25_03.jpeg',1)
    grayscale_1 = cv2.cvtColor(original_1, cv2.COLOR_BGR2GRAY)
    grayscale_2 = cv2.cvtColor(original_2, cv2.COLOR_BGR2GRAY)
    
    print('Image 1')
    plt.axis('off')
    plt.imshow(grayscale_1, cmap=plt.cm.gray)
    plt.show()
    print('Image 2')
    plt.axis('off')
    plt.imshow(grayscale_2, cmap=plt.cm.gray)
    plt.show()
    
    homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
    img_filtered_1 = homo_filter.filter(I=grayscale_1, filter_params=[30,2])
    plt.axis('off')
    print('Image 1 homomorphic filtered')
    plt.imshow(img_filtered_1, cmap=plt.cm.gray)
    plt.show()
    equ_1=cv2.equalizeHist(img_filtered_1)
    plt.axis('off')
    print('Image 1 equalised histogram')
    plt.imshow(equ_1, cmap=plt.cm.gray)
    plt.show()
    
    homo_filter = HomomorphicFilter(a = 0.75, b = 1.25)
    img_filtered_2 = homo_filter.filter(I=grayscale_2, filter_params=[30,2])
    plt.axis('off')
    print('Image 2 homomorphic filtered')
    plt.imshow(img_filtered_2, cmap=plt.cm.gray)
    plt.show()
    equ_2=cv2.equalizeHist(img_filtered_2)
    plt.axis('off')
    print('Image 2 equalised histogram')
    plt.imshow(equ_2, cmap=plt.cm.gray)
    plt.show()
    
    diff=cv2.absdiff(equ_1,equ_2)
        
    #thresholding
    image = diff
    thresh = threshold_otsu(image)
    binary_image = image > thresh
    
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Difference Image')
    ax[0].axis('off')
    
    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')
    
    ax[2].imshow(binary_image, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded Binary image')
    ax[2].axis('off')
    plt.show()
    
    #erosion dilation on binary_image
    binary_image = np.array(binary_image, dtype=np.uint8)
    kernel=np.ones((3,3), np.uint8)
    binary_image = cv2.erode(binary_image, kernel, iterations=2) 
    binary_image = cv2.dilate(binary_image, kernel, iterations=2) 
    plt.axis('off')
    print('Erosion dilation on binary difference image')
    plt.imshow(binary_image, cmap=plt.cm.gray) 
    plt.show()
    
    

    