import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import matplotlib as mpl
from skimage import exposure
import numpy as np
from PIL import Image
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


# Function to define a 2D line
def line2d(x, y, coeffs=[1]*3, return_coeff=False): #coeffs=[1]*3 = [1 1 1]
    a0 = (x*0+1)*coeffs[0]
    a1 = x*coeffs[1]
    a2 = y*coeffs[2]
    if return_coeff:
        return a0, a1, a2
    else:
        return a0+a1+a2

# Function to generate histogram and find highest peaks
def generate_green_histogram(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        pixels = img.load()

        # Get the size of the image
        width, height = img.size

        # Create a list to store green channel values less than 160
        green_values = []

        # Loop through the pixels to get green channel values less than 160
        for y in range(height):
            for x in range(width):
                _, green, _ = pixels[x, y]
                if green < 160:
                    green_values.append(green)

        # Generate a histogram for the green channel values less than 160
        histogram, bin_edges = np.histogram(green_values, bins=range(256))
        sorted_peaks = sorted(range(len(histogram)), key=lambda k: histogram[k], reverse=True)

        # Find the two highest peaks in the histogram with a difference of at least 3
        highest_peaks = []
        for peak_index in sorted_peaks:
            peak_value = bin_edges[peak_index]
            if not highest_peaks or all(abs(peak_value - existing_peak) >= 3 for existing_peak in highest_peaks):
                highest_peaks.append(peak_value)
                if len(highest_peaks) == 2:
                    break

        return max(highest_peaks)

    except Exception as e:
        print("Error:", e)
        return None


# Define the global image variable
global_image = None

# Function to replace a pixel's value
def replace_pixel(x, y, target_value, replacement_value):
    global global_image
    pixel_value = global_image.getpixel((x, y))
    #if pixel_value == target_value:
    global_image.putpixel((x, y), replacement_value)

# Function to identify zones and apply replacements
def identify(mono_l,mono_u,bi_l,bi_u,tri_l,tri_u):
    # Load the images
    global global_image

    global_image = Image.open("saved_bgr.png")
    #zone_mask_img = Image.open("zone_mask.png")

    # Get the dimensions of the images
    width, height = global_image.size

    # Define the target and replacement pixel values
    mono_z1 = 121  # Green channel value (121)
    replacement_value_mono = (201, 255, 52)#(130, 172, 38)  # Replacement value (green)
    replacement_value_bi = (241,183,23)
    replacement_value_tri = (220,110,85)

    # Iterate over each pixel and check the conditions for replacement
    for y in range(height):
        for x in range(width):
            green_channel_value = global_image.getpixel((x, y))[1]  # Green channel value
            #mask_pixel_value = zone_mask_img.getpixel((x, y))

            #mono_l=120
            #mono_u=122

            #bi_l=115 bi_u=118
            #tri_l=112 tri_u=115 (<)

            if mono_l <=green_channel_value<=mono_u: #and mask_pixel_value == (68, 1, 84,255):
                #print("okay")
                replace_pixel(x, y, mono_z1, replacement_value_mono)

            if bi_l <= green_channel_value <=bi_u: #and mask_pixel_value == (68,1,84,255):
              replace_pixel(x,y,mono_z1,replacement_value_bi)

            if tri_l <= green_channel_value < tri_u: #and mask_pixel_value == (68,1,84,255):
              replace_pixel(x,y,mono_z1,replacement_value_tri)

    # Display the modified image
    #global_image.show()
    #display(global_image) THIS
    global_image.show() #WORKING
    #print(global_image)

# Function to reduce background
def reduce_background(img_file, crop):



    ## Image import
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    #print(cv2.imread(img_file))
    ## Bilateral filtering
    imgc = img[crop[0]:crop[1], crop[2]:crop[3]]
    img_bl = (imgc).astype(np.float32)/256
    for ii in range(1):
        img_bl = cv2.bilateralFilter(img_bl,1,1,1)

    bg_px=generate_green_histogram(img_file)
    print(bg_px)
    green_channel = imgc[:, :, 1]
    mask = np.ones(green_channel.shape)
    #mask[green_channel == bg_px] = 0
    mask[(green_channel >= bg_px - 1) & (green_channel <= bg_px + 1)] = 0

    ## Fit to background based on pixels outside the flake.
    y_dim, x_dim, _ = img_bl.shape
    R = img_bl[:,:,0].flatten()
    G = img_bl[:,:,1].flatten()
    #print("G values flattened")
    #print(G) #add
    B = img_bl[:,:,2].flatten()
    X_, Y_ = np.meshgrid(np.arange(x_dim),np.arange(y_dim))
    X = X_.flatten()
    #print("X")
    #print(X)
    Y = Y_.flatten()
    #print("Y")
    #print(Y)
    sub_loc = ((mask.flatten())==0).nonzero()[0] #Finds background pixel indices
    #print("sub_loc")
    #print(sub_loc) #add
    #Rsub = R[sub_loc]
    Gsub = G[sub_loc]
    #print("Gsub")
    #print(Gsub)
    #Bsub = B[sub_loc]
    Xsub = X[sub_loc] #X coordinate of substrate
    #print("Xsub")
    #print(Xsub)
    Ysub = Y[sub_loc] #Y coordinate of substrate
    #print("Ysub")
    #print(Ysub)

    Asub = np.array([*line2d(Xsub, Ysub, return_coeff=True)]).T
    #print("Asub")
    #print(Asub) #add
    #Rcop,_,_,_ = np.linalg.lstsq(Asub, Rsub, rcond=None)
    Gcop,_,_,_ = np.linalg.lstsq(Asub, Gsub, rcond=None)
    #print("Gcop")
    #print(Gcop)
    #Bcop,_,_,_ = np.linalg.lstsq(Asub, Bsub, rcond=None)

    #Rfitp = line2d(X, Y, coeffs=[*Rcop])
    Gfitp = line2d(X, Y, coeffs=[*Gcop])
    #print("Gfitp")
    #print(Gfitp)
    #Bfitp = line2d(X, Y, coeffs=[*Bcop])
    #Rfitp=Gfitp
    #Bfitp=Gfitp

    img_poly = np.dstack([(R-0+1).reshape(y_dim,x_dim)/2,
                          (G-Gfitp+1).reshape(y_dim,x_dim)/2,
                          (B-0+1).reshape(y_dim,x_dim)/2])


    global img_bl2
    img_bl2 = img_poly.astype(np.float32)
    for ii in range(1):
        img_bl2 = cv2.bilateralFilter(img_bl2,1,0.5,1)

    print('Manually inspect background reduction, then close figures.')
    print('img size:',img.shape)
    print('img_bl2 size:',img_bl2.shape)
    plt.imsave("saved_bgr.png", img_bl2,dpi=300)
    plt.imsave("original.png",img,dpi=300)


#Updated get_mode
def get_mode(array):
    # Remove None values from the array
    array = [val for val in array if val is not None]

    if len(array) == 0:
        return None

    #01.11.2023 edit
    #mode_val, counts = mode(array, axis=None)
    unique_elements, counts = np.unique(array, return_counts=True)
    mode_val = unique_elements[counts.argmax()]

    if counts[0] > 0:
        return mode_val[0]
    else:
        return None

args = {'img_file': "adjustustedFocus_tile__ix017_iy018.jpg",
        'crop': [0, -1, 0, -1]}

reduce_background(**args)

#image_prev = np.array(Image.open("saved_bgr.png"))
image=np.floor(np.array(img_bl2)*256)

# Assign zones to the pixels of the image
#zones_array, label_encoder = assign_zones(image)

# Visualize the zones_array as an image

#identify(120,122,115,118,112,115)
#identify(121,124,115,121,113,118) #old_pink
identify(119,122,107,118,105,107) #old_green

fig, axs = plt.subplots()
fig.subplots_adjust(bottom=0.25)

im = axs.imshow(global_image)
