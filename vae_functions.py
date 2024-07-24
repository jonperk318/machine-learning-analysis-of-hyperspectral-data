# UNSUPERVISED VAE

import matplotlib as mpl
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300
from matplotlib.gridspec import GridSpec
import numpy as np

save_directory = "./svg-figures/" # modify to save figures as a specified file extension
file_extension = ".svg"

# Plotting pixels containing entire spectra in a latent space and corresponding images
def vae_plot(suptitle, z_mean, xpix, ypix, scalebar_size, save=False):
    
    rows = 2
    columns = len(z_mean[0])
    
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(suptitle, fontsize=10)
    fig.patch.set_facecolor("white")
    
    for i in range(rows):
        for j in range(columns):
            
            if i == 0: # First row
                
                if columns == 2: # 2D Latent space
                    ax = fig.add_subplot(2, 2, j + 1)
                    ax.scatter(z_mean[:, 0], z_mean[:, 1], c=z_mean[:, j], cmap=cmap,
                            ec='black', lw=0.1, alpha=0.6, s=2)
                    ax.tick_params(axis='both', which='major', labelsize=4, direction='out', 
                               length=2, width=0.5)
                    
                else: # 3D Latent space
                    ax = fig.add_subplot(2, 3, j + 1, projection='3d')
                    ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], 
                            c=z_mean[:, j], cmap=cmap,
                            ec='black', lw=0.05, alpha=0.5, s=2)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.zaxis.set_ticklabels([])
                    ax.set_xlabel("X", size=5, labelpad=-15)
                    ax.set_ylabel("Y", size=5, labelpad=-15)
                    ax.set_zlabel("Z", size=5, labelpad=-15)
                    if j == 0:
                        ax.view_init(30, -85) 
                    elif j == 1:
                        ax.view_init(30, -40)
                    elif j == 2:
                        ax.view_init(30, 5)
                
                if j == 0:
                    ax.set_title('Latent Space', size=7)
            
            if i == 1: # Second row
                
                if columns == 2: # 2D Latent Space
                    img = fig.add_subplot(2, 2, j+3)
                else: # 3D Latent Space
                    img = fig.add_subplot(2, 3, j+4)
                    
                img.imshow(z_mean[:,j].reshape(ypix, xpix), cmap=cmap)
                img.set_xticks([])
                img.set_yticks([])
                
                if j == (columns-1): # Add scalebar to bottom right image
                    scalebar = AnchoredSizeBar(img.transData, scalebar_size, " ", 
                            "lower right", pad=0.2, color='#F2F2F2', frameon=False,
                                               size_vertical=3, label_top=True)
                    img.add_artist(scalebar)
    
    if save:
        fig.savefig(save_directory + "unsupervised-vae/" + suptitle + file_extension)
        
# Spectra Plot (for determining latent dimensions)
def spectra_plot(suptitle, img, array, xpix, ypix, wav, save=False):
    
    n = xpix*ypix # total number of spectra
    cmap = mpl.colormaps['viridis'](np.linspace(0,1,n)) # create colormap
    fig = plt.figure(figsize=(6,3))
    fig.suptitle(suptitle, fontsize=10)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot()
    ax.set_prop_cycle('color', list(cmap)) # colormaps each spectra from first to last in array
    ax.tick_params(axis='both', which='major', labelsize=5, direction='out', length=2, width=0.5)
    plt.xlim(short_wav - 5, long_wav + 5)
    #ax.set_xticks([500, 750])
    #ax.set_xticklabels([])
    #plt.yscale('log')
    
    for i in array:
        ax.plot(wav, img[:,i[0],i[1]], lw=0.03, alpha=0.5)
    
    if save:
        fig.savefig(save_directory + "unsupervised-vae/" + suptitle + file_extension)

        
# SEMI-SUPERVISED VAE

# Plot the latent space and image reconstruction of SSVAE models
def ssvae_plot(suptitle, z_mean, z_labels, xpix, ypix, scalebar_size, save=False):
    
    rows = 2
    columns = 1
    
    cmap = mpl.colormaps['viridis']
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(suptitle, fontsize=10)
    gs = GridSpec(rows, columns)
    #fig.patch.set_facecolor('#00000000')
    fig.patch.set_facecolor('white')
    
    for i in range(rows):
        column_count = 0
        for j in range(columns):
            if i == 0:
                ax = fig.add_subplot(gs[0,column_count])
                ax.scatter(z_mean[:,0], z_mean[:,1], c=z_labels, cmap=cmap,
                        ec='black', lw=0.05, alpha=0.6, s=2)
                #ax.tick_params(axis='both', labelsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
            if i == 1:
                img = fig.add_subplot(gs[1,column_count])
                img.imshow(z_labels.reshape(ypix, xpix), cmap=cmap)
                img.set_xticks([])
                img.set_yticks([])
                
                if column_count == (columns-1):
                    scalebar = AnchoredSizeBar(img.transData, scalebar_size, " ", 
                            "lower right", pad=0.3, color='#F2F2F2', frameon=False,
                                               size_vertical=3, label_top=True)
                    img.add_artist(scalebar)

            column_count += 1
    
    if save:
        fig.savefig(save_directory + "semi-supervised-vae/" + suptitle + file_extension)

# Function to plot every spectrum with a certain x value (along a vertical line in the image)
def pointfinder_plot(img, x_pt, ypix, wav):
    
    rows = ypix
    columns = 1
    fig = plt.figure(figsize=(4, rows))
    
    for i in range(rows):

        ax = fig.add_subplot(rows, columns, i + 1)
        ax.plot(wav, img[:, i, x_pt], c=color[i], lw=2)
        ax.text(220, 350, str([i, x_pt]), fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])   

    plt.show()

# Plot all spectra in each of the 4 classes
def spectsig_plot(suptitle, img, pts, wav, save=False):
    
    rows = len(pts[0])
    columns = 4
    gs = GridSpec(rows, columns)
    gs.update(wspace=0, hspace=0)

    figsize = (columns*2, rows)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.suptitle(suptitle, fontsize=20)
    #fig.subplots_adjust(top=0.95, bottom=0.1)

    row_count = 0
    for i in range(rows):

        column_count = 0
        for j in range(columns):

            fig_plt = fig.add_subplot(gs[i,j])
            fig_plt.plot(wav, img[pts[j][i][0],pts[j]
                                  [i][1],:], c=cmap_list[j], lw=1)
            plt.xlim(short_wav - 5, long_wav + 5) # Cropping the spectra shown in the plot
            #fig_plt.tick_params(axis='both', direction='out', length=12, width=4)
            fig_plt.set_xticks([])
            fig_plt.set_yticks([])
            
            #if column_count == 0:
                #fig_plt.set_yticks([1000])
                #fig_plt.set_yticklabels([])
                #fig_plt.tick_params(axis='y', direction='out', length=12, width=4)
            
            #if row_count == rows-1:
                #fig_plt.set_xticks([400-short_wav, 750-long_wav])
                #fig_plt.set_xticklabels([])
                #fig_plt.tick_params(axis='x', direction='out', length=12, width=4)

            column_count += 1

        row_count += 1
    
    if save:
        fig.savefig(save_directory + "semi-supervised-vae/" + suptitle + file_extension)
    plt.show()
    
# This plot shows the first 180 spectra the SSVAE model assigns to a specified class
def ssvae_label_plot(img, pts, wav, starting_index=0, save=False):
    
    rows = 180
    #rows = len(pts)
    columns = 1
    fig = plt.figure(figsize=(4, rows))
    fig.patch.set_facecolor("white")
    #fig.suptitle(suptitle, fontsize=30)
    gs = GridSpec(rows, columns)
    
    row_count = 0
    for i in range(starting_index, starting_index + rows):

        fig_plt = fig.add_subplot(gs[row_count,0])
        fig_plt.plot(wav, img[:, pts[i,0], pts[i,1]], 
                     c=color[row_count], lw=3)
        plt.xticks([])
        plt.yticks([])
        #plt.text(220,350,str((int(pts[i][0]),int(pts[i][1]))))
        plt.text(0, 0, str(i))
        row_count += 1

    #gs.tight_layout(fig, rect=[0,0.03,1,0.95])
    if save:
        fig.savefig(save_directory + "semi-supervised-vae/" + suptitle + file_extension)
    plt.show()

# Plot one spectrum from each class in the labeled data
def spect_label_plot(suptitle, img, pts, wav, save=False):
    
    rows = 1
    columns = 4
    gs = GridSpec(rows, columns)
    gs.update(wspace=0,hspace=0.2)

    figsize = (columns*4, rows*3)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle, size = 20)
    fig.patch.set_facecolor("white")
    #fig.patch.set_facecolor('#00000000')
    #fig.suptitle(suptitle, fontsize=30)
    #fig.subplots_adjust(top=0.95, bottom=0.1)

    row_count = 0
    for i in range(rows):

        column_count = 0
        for j in range(columns):

            fig_plt = fig.add_subplot(gs[i,j])
            fig_plt.plot(wav, img[:,pts[j][i][0],pts[j][i][1]], c=cmap_list[j], lw=3)
            #plt.text(300, 370, str(column_count))
            #fig_plt.set_title(method_names[row_count] + point_names[column_count])
            #fig_plt.tick_params(color=color[j])
            plt.xlim(short_wav - 5, long_wav + 5) # Cropping the spectra shown in the plot
            #fig_plt.tick_params(axis='both', direction='out', length=12, width=4)
            #fig_plt.set_xticks([400, 550, 750])
            fig_plt.set_yticks([])
            fig_plt.set_xticks([])

            column_count += 1

        row_count += 1
    
    if save:
        fig.savefig(save_directory + "/semi-supervised-vae/" + suptitle + file_extension)
    plt.show()

# Plot one spectrum from each class in the latent representation of the data
def custom_spect_label_plot(suptitle, img, pts, wav, save=False):
    
    rows = 1
    columns = 4
    gs = GridSpec(rows, columns)
    gs.update(wspace=0,hspace=0.2)

    figsize = (columns*4, rows*3)
    fig = plt.figure(figsize=figsize)
    #fig.patch.set_facecolor("#00000000")
    fig.patch.set_facecolor("white")
    fig.suptitle(suptitle, fontsize=20)
    #fig.subplots_adjust(top=0.95, bottom=0.1)

    row_count = 0
    for i in range(rows):

        column_count = 0
        for j in range(columns):

            fig_plt = fig.add_subplot(gs[i, j])
            fig_plt.plot(wav, img[:, pts[j][0], pts[j][1]], c=cmap_list[j], lw=3)
            #plt.text(300, 370, str(column_count))
            #fig_plt.set_title(method_names[row_count] + point_names[column_count])
            #fig_plt.tick_params(color=color[j])
            plt.xlim(short_wav, long_wav) # Cropping the spectra shown in the plot
            fig_plt.tick_params(axis='both', direction='out', length=6, width=2)
            #fig_plt.set_xticks([400, 750])
            fig_plt.set_xticks([])
            fig_plt.set_yticks([])
            #fig_plt.set_xticklabels([])
            
            #if column_count == 0:
                #fig_plt.set_yticks([1000])
                #fig_plt.set_yticklabels([])
                #fig_plt.tick_params(axis='y', direction='out', length=12, width=4)
            
            #if row_count == rows-1:
                #fig_plt.set_xticks([400-short_wav, 750-long_wav])
                #fig_plt.set_xticklabels([])
                #fig_plt.tick_params(axis='x', direction='out', length=12, width=4)

            column_count += 1

        row_count += 1
    
    if save:
        fig.savefig(save_directory + "/semi-supervised-vae/" + suptitle + file_extension)
    plt.show()