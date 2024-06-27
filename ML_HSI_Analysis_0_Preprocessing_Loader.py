### Imports
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation, ArtistAnimation
from celluloid import Camera
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from typing import Optional, Union, List, Tuple
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import find_peaks, savgol_filter
from skimage.restoration import denoise_wavelet


### Functions and Classes
def cropper(wavelength_file, spectra_file, shortwavelength, longwavelength):
    
    # This function allows us to crop all wavelengths below and above a specified value
    # Returns new wavelengths object and new spectra file
    
    if (
        shortwavelength < wavelength_file.min()
        or longwavelength > wavelength_file.max()
    ):
        print(
            "Desired wavelength exceeds range. Choose a value between {}nm and {}nm".format(
                wavelength_file.min(), wavelength_file.max()
            )
        )
        return

    y = []

    shortwavelengthindex = find_nearest(wavelength_file, shortwavelength)
    longwavelengthindex = find_nearest(wavelength_file, longwavelength)

    x = wavelength_file[shortwavelengthindex:longwavelengthindex]

    for i in range(len(spectra_file[shortwavelengthindex:longwavelengthindex][:])):
        y.append(spectra_file[shortwavelengthindex:longwavelengthindex][i])
        
    y = np.asarray(y)
    z = longwavelengthindex - shortwavelengthindex
        
    return x, y, z


def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx


def normalize(array):
    return (array - array.min()) / (array.max() - array.min())


class data:
    
    def __init__(self, datapath):
        self.h5 = h5py.File(datapath, "r")
        self.meas0 = acquisition(self.h5["Acquisition0"])
        self.meas1 = acquisition(self.h5["Acquisition1"])
        self.meas2 = acquisition(self.h5["Acquisition2"])
        self.meas3 = acquisition(self.h5["Acquisition3"])
        return
    
    def close(self):
        return self.h5.close()
    

class acquisition:
    
    def __init__(self, acquisition):
        self.name = acquisition["PhysicalData"]["ChannelDescription"][()][0]
        self.xdim = acquisition["ImageData"]["DimensionScaleX"][()]
        self.ydim = acquisition["ImageData"]["DimensionScaleY"][()]
        self.zdim = acquisition["ImageData"]["DimensionScaleZ"][()]
        self.image = acquisition["ImageData"]["Image"][()].squeeze()
        try:
            self.xpix = self.image.shape[0]
            self.ypix = self.image.shape[1]
        except IndexError:
            self.xpix = self.image.shape[0]
            self.ypix = 0
        self.acqname = acquisition.name
        if self.acqname == "/Acquisition2":
            self.wavelengths = acquisition["ImageData"]["DimensionScaleC"][()] * 1e9
            try:
                self.xpix = self.image.shape[2]
                self.ypix = self.image.shape[1]
                self.zpix = self.image.shape[0]
            except IndexError:
                self.xpix = self.image.shape[1]
                self.ypix = 0
        return
    
    def denoise(self, subtractbackground=False, minusmedian=False):
        if self.acqname != "/Acquisition2":
            print("Can only denoise CL, please choose CL measure.")
            pass
        denoise_kwargs = dict(
            multichannel=False,
            convert2ycbcr=False,
            wavelet="sym8",
            rescale_sigma=False,
            wavelet_levels=6,
            mode="soft",
        )
        if self.ypix == 0:
            spectra = self.image.transpose(1, 0)
            spectra = [savgol_filter(s, 5, 2) for s in spectra]
            spectra = [median_filter(s, size=(5)) for s in spectra]
            if subtractbackground==True:
                spectra = [s-np.mean(s[0:200]) for s in spectra]
                spectra = [s.clip(min=0) for s in spectra]
            return np.reshape(spectra, (self.xpix, self.image.shape[0]))
        else:
            spectra = self.transposedata().reshape(
                self.xpix * self.ypix, self.image.shape[0]
            )
            denoise_kwargs = dict(
                multichannel=False,
                convert2ycbcr=False,
                wavelet="sym8",
                rescale_sigma=False,
                wavelet_levels=6,
                mode="soft",
            )
            spectra = [savgol_filter(s, 5, 2) for s in spectra]
            spectra = [median_filter(s, size=(5)) for s in spectra]
            if subtractbackground==True:
                spectra = [s-np.mean(s[0:200]) for s in spectra]
                spectra = [s.clip(min=0) for s in spectra]
            return np.reshape(spectra, (self.xpix, self.ypix, self.image.shape[0]))

        
### Colors
random_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(100)]
color = ['#3a86ff', '#ffbe0b', '#ff006e', '#8338ec', '#fb5607', '#390099'] * 1000


### Reading Data
filepath1 = "20220507_MAPbI3Grain1_5kV_0p11nA_2kx.nosync.h5"
filepath2 = "20220508_MAPbI3Grain5EdgeMap_5kV_14pA_16kx.nosync.h5"
#filepath1 = "20220507_MAPbI3Grain1_5kV_0p11nA_2kx.h5"
#filepath2 = "20220508_MAPbI3Grain5EdgeMap_5kV_14pA_16kx.h5"

ds1 = data(filepath1)
ds2 = data(filepath2)

# Wavelength values (1024 wavelengths in the z dimension)
f1_wav = ds1.meas2.wavelengths
f2_wav = ds2.meas2.wavelengths

f1_wav_orig = f1_wav
f2_wav_orig = f1_wav

# Grain 1: Wide-field, low resolution hybrid perovskite (HP) grain
f1 = data(filepath1).h5

# Grain 2: High-resolution HP grain boundary
f2 = data(filepath2).h5


f1_img1 = acquisition(f1["Acquisition1"]).image
f1_img2 = acquisition(f1["/Acquisition2"]).image

f1_xpix = acquisition(f1["/Acquisition2"]).xpix
f1_ypix = acquisition(f1["/Acquisition2"]).ypix
f1_zpix = acquisition(f1["/Acquisition2"]).zpix

f1_zpix_orig = f1_zpix

f1_img2_2d = np.reshape(f1_img2, (f1_zpix, f1_ypix*f1_xpix))

f2_img1 = acquisition(f2["Acquisition1"]).image
f2_img2 = acquisition(f2["/Acquisition2"]).image

f2_xpix = acquisition(f2["/Acquisition2"]).xpix
f2_ypix = acquisition(f2["/Acquisition2"]).ypix
f2_zpix = acquisition(f2["/Acquisition2"]).zpix

f2_zpix_orig = f2_zpix

f2_img2_2d = np.reshape(f2_img2, (f2_zpix, f2_ypix*f2_xpix))

# Bottom and top of range for spectra
short_wav = 350
long_wav = 900

f1_wav, f1_img2_cropped, f1_zpix = cropper(f1_wav, f1_img2_2d, short_wav, long_wav)
f2_wav, f2_img2_cropped, f2_zpix = cropper(f2_wav, f2_img2_2d, short_wav, long_wav)

f1_img2 = np.reshape(f1_img2_cropped, (f1_zpix, f1_ypix, f1_xpix))
f2_img2 = np.reshape(f2_img2_cropped, (f2_zpix, f2_ypix, f2_xpix))

f1_pix = f1_zpix * f1_ypix * f1_xpix
f2_pix = f2_zpix * f2_ypix * f2_xpix


### Cosmic Rays
def cosmic_ray_finder_1(img, jump1): # Image dimension
    # Identifies some pixels affected by cosmic rays and returns a list of points
    
    pts = []
    
    for z in range(len(img)):
        for y in range(len(img[z])):
            for x in range(len(img[z][y])):
                
                if x == 0: # Left edge
                    
                    if y == 0: # Top left corner
                        if (img[z,y,x] > jump1*(img[z,y,x+1]) 
                            or img[z,y,x] > jump1*(img[z,y+1,x])):
                            pts.append([z,y,x])
                    elif y == (len(img[z])-1): # Bottom left corner
                        if (img[z,y,x] > jump1*(img[z,y,x+1]) 
                            or img[z,y,x] > jump1*(img[z,y-1,x])):
                            pts.append([z,y,x])
                    else:
                        if (img[z,y,x] > jump1*(img[z,y,x+1]) or img[z,y,x] > jump1*(img[z,y-1,x]) 
                            or img[z,y,x] > jump1*(img[z,y+1,x])):
                            pts.append([z,y,x])
                        
                elif x == (len(img[z][y])-1): # Right edge
                
                    if y == 0: # Top right corner
                        if (img[z,y,x] > jump1*(img[z,y,x-1]) 
                            or img[z,y,x] > jump1*(img[z,y+1,x])):
                            pts.append([z,y,x])
                    elif y == (len(img[z])-1): # Bottom right corner
                        if (img[z,y,x] > jump1*(img[z,y,x-1]) 
                            or img[z,y,x] > jump1*(img[z,y-1,x])):
                            pts.append([z,y,x])
                    else:
                        if (img[z,y,x] > jump1*(img[z,y,x-1]) or img[z,y,x] > jump1*(img[z,y-1,x]) 
                            or img[z,y,x] > jump1*(img[z,y+1,x])):
                            pts.append([z,y,x])
                        
                elif y == 0: # Top edge (not including corners)
                    
                    if (img[z,y,x] > jump1*(img[z,y,x-1]) or img[z,y,x] > jump1*(img[z,y,x+1]) 
                        or img[z,y,x] > jump1*(img[z,y+1,x])):
                        pts.append([z,y,x])
                        
                elif y == (len(img[z])-1): # Bottom edge (not including corners)
                    
                    if (img[z,y,x] > jump1*(img[z,y,x-1]) or img[z,y,x] > jump1*(img[z,y,x+1]) 
                        or img[z,y,x] > jump1*(img[z,y-1,x])):
                        pts.append([z,y,x])
                        
                else:
                    
                    if (img[z,y,x] > jump1*(img[z,y,x-1]) or img[z,y,x] > jump1*(img[z,y,x+1])
                       or img[z,y,x] > jump1*(img[z,y-1,x]) or img[z,y,x] > jump1*(img[z,y+1,x])):
                        pts.append([z,y,x])
                    
    return pts

f1_cosmic_pts_1 = cosmic_ray_finder_1(f1_img2, 7)
f2_cosmic_pts_1 = cosmic_ray_finder_1(f2_img2, 7)

def cosmic_ray_remover(img, zpix, cosmic_pts, radius=5):
    
    wavs_changed = 0 # Number of wavelengths in the array changed
    
    for pt in cosmic_pts:
        
        bottom = radius
        top = radius
        
        while bottom >= 0: # making sure bottom of range isn't less than 0
            if pt[0]-bottom >= 0:
                range_bottom = pt[0]-bottom
                break
            bottom -= 1
            
        while top >= 0: # making sure top of range isn't greater than length of z dimension
            if pt[0]+top <= zpix:
                range_top = pt[0]+top
                break
            top -= 1
        
        for z in range(range_bottom, range_top):
            
            try: pix1 = img[z,pt[1],pt[2]-1]
            except: pix1 = img[z,pt[1],pt[2]+1]
            try: pix2 = img[z,pt[1],pt[2]+1]
            except: pix2 = img[z,pt[1],pt[2]-1]
            try: pix3 = img[z,pt[1]-1,pt[2]]
            except: pix4 = img[z,pt[1]+1,pt[2]]
            try: pix4 = img[z,pt[1]+1,pt[2]]
            except: pix4 = img[z,pt[1]-1,pt[2]]
            try: pix5 = img[z,pt[1]-1,pt[2]-1]
            except: pix5 = img[z,pt[1]+1,pt[2]+1]
            #except: pix5 = img[z,pt[1]-1,pt[2]+1]
            try: pix6 = img[z,pt[1]-1,pt[2]+1]
            except: pix6 = img[z,pt[1]+1,pt[2]-1]
            #except: pix6 = img[z,pt[1]-1,pt[2]-1]
            try: pix7 = img[z,pt[1]+1,pt[2]-1]
            except: pix7 = img[z,pt[1]-1,pt[2]+1]
            #except: pix7 = img[z,pt[1]+1,pt[2]+1]
            try: pix8 = img[z,pt[1]+1,pt[2]+1]
            except: pix8 = img[z,pt[1]-1,pt[2]-1]
            #except: pix8 = img[z,pt[1]+1,pt[2]-1]
                
            img[z,pt[1],pt[2]] = np.mean((pix1, pix2, pix3, pix4, pix5, pix6, pix7, pix8))
            
            wavs_changed += 1
        
    return img, wavs_changed

f1_img2, f1_wavs_changed_1 = cosmic_ray_remover(f1_img2, f1_zpix, f1_cosmic_pts_1)
f2_img2, f2_wavs_changed_1 = cosmic_ray_remover(f2_img2, f2_zpix, f2_cosmic_pts_1)

def cosmic_ray_finder_2(img, jump2): # Spectral dimension
    # Identifies the rest of the pixels affected by cosmic rays and returns a list of points
    
    pts = []
    
    for z in range(len(img)):
        for y in range(len(img[z])):
            for x in range(len(img[z][y])):
                
                if z == 0: # First wavelength
                    if (img[z,y,x] > jump2*(img[z+1,y,x])):
                        pts.append([z,y,x])
                        
                elif z == (len(img)-1): # Last wavelength
                    if (img[z,y,x] > jump2*(img[z-1,y,x])):
                        pts.append([z,y,x])
                        
                else:
                    if (img[z,y,x] > jump2*(img[z-1,y,x]) or img[z,y,x] > jump2*(img[z+1,y,x])):
                        pts.append([z,y,x])
                    
    return pts

f1_cosmic_pts_2 = cosmic_ray_finder_2(f1_img2, 2)
f2_cosmic_pts_2 = cosmic_ray_finder_2(f2_img2, 2)

f1_img2, f1_wavs_changed_2 = cosmic_ray_remover(f1_img2, f1_zpix, f1_cosmic_pts_2)
f2_img2, f2_wavs_changed_2 = cosmic_ray_remover(f2_img2, f2_zpix, f2_cosmic_pts_2)


### Reshaping
f1_img2_1d = np.reshape(f1_img2, (f1_zpix*f1_ypix*f1_xpix), order='C')
f2_img2_1d = np.reshape(f2_img2, (f2_zpix*f2_ypix*f2_xpix), order='C')

f1_img2_2d = np.reshape(f1_img2, (f1_zpix, f1_ypix*f1_xpix), order='C')
f2_img2_2d = np.reshape(f2_img2, (f2_zpix, f2_ypix*f2_xpix), order='C')


### Initial Plots
f1_x_points = [50, 10, 142, 53]
f1_y_points = [50, 65, 22, 89]

f2_x_points = [70, 10, 58, 55]
f2_y_points = [10, 67, 40, 40]

def initial_plot(suptitle, img1, img2, spect_pt, x_points, y_points, wav, 
                 scalebar_size, save=False):
    
    fig = plt.figure(figsize=(18, 4))
    rows = 1
    columns = 3
    gs = GridSpec(rows, columns)
    gs.update(wspace=0.1,hspace=0.1, top=0.75)

    fig.suptitle(suptitle, fontsize=30)
    #fig.patch.set_facecolor('#00000000')

    # img1
    fig_img1 = fig.add_subplot(gs[0,0])
    fig_img1.imshow(img1[:,:], cmap='gray')
    fig_img1.set_title("SEM Image")
    fig_img1.set_xticks([])
    fig_img1.set_yticks([])

    # img2
    fig_img2 = fig.add_subplot(gs[0,1])
    fig_img2.imshow(img2[spect_pt-short_wav,:,:])
    fig_img2.set_title("CL Image")
    fig_img2.set_xticks([])
    fig_img2.set_yticks([])
    scalebar = AnchoredSizeBar(fig_img2.transData, scalebar_size, " ", "lower right",
                           pad=0.3,
                           color='#F2F2F2',
                           frameon=False,
                           size_vertical=3,
                           label_top=True)
    fig_img2.add_artist(scalebar)
    
    count = 0
    for i in range(len(x_points)):
        fig_img2.plot(x_points[count], y_points[count], "o", ms=10, c=color[i])
        count += 1

    # plt
    fig_plt = fig.add_subplot(gs[0,2:])
    fig_plt.set_title("Spectra")
    #fig_plt.set_xticks([400-short_wav, 750-long_wav])
    #fig_plt.set_yticks([400, 800])
    #fig_plt.set_xticklabels([])
    #fig_plt.set_yticklabels([])
    plt.xlim(short_wav-5, long_wav+5)
    fig_plt.tick_params(axis='both', direction='out', length=8, width=2)
    
    count = 0
    for i in range(len(x_points)):
        fig_plt.plot(wav, img2[:, y_points[count], x_points[count]], 
                     c=color[i], lw=3)
        #print(np.shape(img2[:, y_points[count], x_points[count]]))
        count += 1

    fig_plt.axvline(x=spect_pt-short_wav, c='black',lw=1, linestyle=':')
    
    if save:
        fig.savefig("Preprocessing/" + suptitle + ".png")
    plt.show()


### Denoising
# Grain 1
f1_sb_mean = np.mean(f1_img2_1d[0:100])
f1_sub_back = [(i - f1_sb_mean) for i in f1_img2_1d]
f1_sub_back = [i.clip(min=0) for i in f1_sub_back]
f1_sub_back = np.reshape(f1_sub_back, (f1_zpix, f1_ypix, f1_xpix), order='C')

f1_sub_back_1d = np.reshape(f1_sub_back, (f1_zpix*f1_ypix*f1_xpix), order='C')
f1_sb_median = median_filter(
    f1_sub_back_1d, size=3, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
f1_sb_median = np.reshape(f1_sb_median, (f1_zpix, f1_ypix, f1_xpix), order='C')

# Grain 2
f2_sb_mean = np.mean(f2_img2_1d[0:100])
f2_sub_back = [(i - f2_sb_mean) for i in f2_img2_1d]
f2_sub_back = [i.clip(min=0) for i in f2_sub_back]
f2_sub_back = np.reshape(f2_sub_back, (f2_zpix, f2_ypix, f2_xpix), order='C')

f2_sub_back_1d = np.reshape(f2_sub_back, (f2_zpix*f2_ypix*f2_xpix), order='C')
f2_sb_median = median_filter(
    f2_sub_back_1d, size=2, footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
f2_sb_median = np.reshape(f2_sb_median, (f2_zpix, f2_ypix, f2_xpix), order='C')

f1_denoised_2d = np.reshape(f1_sb_median, (f1_zpix, f1_ypix*f1_xpix), order='C')
f2_denoised_2d = np.reshape(f2_sb_median, (f2_zpix, f2_ypix*f2_xpix), order='C')


### Important objects
print("SEM images: f1_img1, f2_img1")
print("CL images: f1_img2, f2_img2")
print("Denoised data: f1_sb_median, f2_sb_median")
print("2D denoised data: f1_denoised_2d, f2_denoised_2d")
print("Example points: f1_x_points, f1_y_points, f2_x_points, f2_y_points")
print("Wavelengths and dimensions: f1_wav, f2_wav, " 
      "f1_xpix, f1_ypix, f1_zpix, f2_xpix, f2_ypix, f2_zpix")