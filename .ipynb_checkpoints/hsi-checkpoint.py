# Imports

#import numpy as np
#import PIL
#from PIL  import ImageFilter
#from PIL import Image
#import matplotlib.pyplot as plt
#from scipy.special import wofz


# Functions

def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

def gaussian(x,amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def lorentzian(x,amp1,cen1,wid1):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2))

def voigt(x,x0,y0,a,sig,gam):
    return y0 + a * np.real(wofz((x-x0 + 1j*gam) / sig / np.sqrt(2))) / sig / np.sqrt(2 * np.pi)


# Classes

class HSI:
    
    # HSI is created at initialization step
   
    def __init__(self, img, dist1, dist2, dist3, zpix, ypix, xpix,
                filter_type=None, filter_strength=0, noise=1):
      
        self.filter_type = filter_type
        self.filter_strength = filter_strength
      
        # Adding image filter
        if self.filter_type == 'Gaussian' or 'gaussian':
            self.im = img.filter(ImageFilter.GaussianBlur(radius = self.filter_strength))
        if self.filter_type == 'Box' or 'box':
            self.im = img.filter(ImageFilter.BoxBlur(radius = self.filter_strength))
        if self.filter_type == None or 'None' or 'none':
            self.im = img
      
        # Initializing noise type
        if noise == 1:
            self.noise = np.random.normal(0,.37,zpix)
        if noise == 2:
            self.noise = np.random.normal(0,1.85,zpix)
        if noise ==3 :
            self.noise = np.random.normal(0,3.5,zpix)
        
        # Adding noise to distributions
        self.dist1 = dist1 + self.noise
        self.dist2 = dist2 + self.noise
        self.dist3 = dist3 + self.noise
      
        # Creating the hsi
        r, g, b = self.im.split()
        zero = r.point(lambda _ : 0) # Creates xpix by ypix array of zeros

        red_merge = Image.merge("RGB", (r, zero, zero))
        green_merge = Image.merge("RGB", (zero, g, zero))
        blue_merge = Image.merge("RGB", (zero, zero, b))
        red_array = np.asarray(red_merge)[:,:,0]
        green_array = np.asarray(green_merge)[:,:,1]
        blue_array = np.asarray(blue_merge)[:,:,2]
        red = normalize(red_array.flatten()).tolist()
        blue = normalize(blue_array.flatten()).tolist()
        green = normalize(green_array.flatten()).tolist()

        for i in range(len(red)):
            red[i] = red[i] * self.dist1
        for i in range(len(green)):
            green[i] = green[i] *self.dist2
        for i in range(len(blue)):
            blue[i] = blue[i] * self.dist3

        hyper = []
        for i in range(len(blue)):
            idx = red[i] + green[i] + blue[i]
            hyper.append(idx)

        hyper = np.asarray(hyper).reshape(xpix, ypix, zpix)
        hyper = np.swapaxes(hyper, 0, 2)
        hyper = np.rot90(hyper, k=3, axes=(1, 2))
        hyper = np.flip(hyper, axis=2)
        self.hyper = hyper
              
        
    # Original image  
    def img(self):
       plt.imshow(self.im)
    
    
    # Image at certain band
    def hyper_img(self,band):
        return(plt.imshow(self.hyper[:,:,band],cmap = 'viridis'))


   # Spectra at certain pixel
    def hyper_band(self,x,y,color):
        self.color = str(color)
        return (plt.plot(self.hyper[:,y,x],color = self.color))