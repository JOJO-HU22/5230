from PIL import Image, ImageDraw, ImageEnhance
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy
import torch
import torchvision.models as models
import os
import cv2
from models import *
import warnings
import shutil

from utils import rotate
from utils import stick
from utils import mapping3d
from utils import feature
from concurrent.futures import ThreadPoolExecutor, TimeoutError  # Module for timeout control
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filtering to reduce image noise.
   :param image: input image (PIL format)
   :param diameter: diameter in pixels of the filter
   :param sigma_color: standard deviation in color space
   :param sigma_space: standard deviation in coordinate space
   :return: processed image
    """
    # Convert the image to a NumPy array
    image_np = np.array(image)
    # Apply bilateral filtering
    filtered_image = cv2.bilateralFilter(image_np, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # Convert back to PIL image format
    return Image.fromarray(filtered_image)


# FGSM attack function
def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM Attack: Generate perturbed image based on the gradient of the loss with respect to the input image.
    :param image: Original image
    :param epsilon: Perturbation magnitude
    :param data_grad: Gradient of the loss with respect to the input image
    :return: Perturbed image
    """
    sign_data_grad = data_grad.sign()  # Get the sign of the gradient
    perturbed_image = image + epsilon * sign_data_grad  # Add perturbation
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Clip to valid image range
    return perturbed_image
# Function to add global noise
def add_global_noise(image, noise_factor=0.1):
    """
    Add global noise to the entire image
    :param image: original PIL image
    :param noise_factor: noise intensity, the larger the value, the stronger the noise
    :return: the image after adding noise
    """
    noise = np.random.normal(loc=0, scale=noise_factor, size=(image.height, image.width, 3))
    noisy_image = np.array(image) / 255.0 + noise
    noisy_image = np.clip(noisy_image, 0, 1) * 255
    return Image.fromarray(noisy_image.astype('uint8'))

# Semantic attack function
def semantic_attack(image, brightness_factor=1.1, contrast_factor=1.2, color_factor=1.01):
    """
    Semantic attack: Generate adversarial samples by adjusting the brightness, contrast and color of the image
    :param image: original PIL image
    :param brightness_factor: brightness adjustment factor (1.0 means unchanged)
    :param contrast_factor: contrast adjustment factor (1.0 means unchanged)
    :param color_factor: color adjustment factor (1.0 means unchanged)
    :return: image after semantic attack
    """
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Adjust Color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    return image
# Prediction function with FGSM-based and gray-box adversarial perturbation
def predict_type_facenet(image_perturbed, cleancrop, true_label):
    """
    Perform predictions and generate adversarial examples using FGSM.
    :param image_perturbed: input image
    :param cleancrop: MTCNN candidate image
    :param true_label: true label, used to generate adversarial examples
    :return: prediction results of adversarial examples and perturbed images
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model
    resnet = torch.load('/home/featurize/work/common/resnet_model.pth', map_location='cuda:0').to(device)
    resnet.eval()  # Set to evaluation mode
    resnet.classify = True

    # Convert labels to tensors and pass them to the device
    label = torch.tensor([true_label]).to(device)

    # The disturbance intensity of FGSM attack, increasing the disturbance amplitude
    epsilon = 0.3  # Controlling the disturbance amplitude
    # Set requires_grad=True for the image to compute gradients
    image_perturbed.requires_grad = True

    # Forward Propagation
    output = resnet(image_perturbed)
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function
    loss = loss_fn(output, label)    # Calculating Losses

    # Backpropagation to calculate gradients
    resnet.zero_grad()
    loss.backward()

    # Get the gradient of an image
    image_grad = image_perturbed.grad.data

    # Detecting facial bounding boxes using MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(image_perturbed)

    # If a face is detected, use the first detected face for perturbation
    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])  # Get bounding box coordinates
    else:
        # If no face is detected, fall back to the original image size
        x1, y1, x2, y2 = 0, 0, image_perturbed.size(2), image_perturbed.size(1)

    # Create a mask of the face area
    mask = torch.zeros_like(image_perturbed)
    mask[:, :, y1:y2, x1:x2] = 1  # Apply perturbations only in the face region

    # Generating Adversarial Examples
    perturbed_image = fgsm_attack(image_perturbed, epsilon, image_grad)
    perturbed_image[:, :, y1:y2, x1:x2] = image_perturbed[:, :, y1:y2, x1:x2] + epsilon * mask[:, :, y1:y2, x1:x2] * image_grad.sign()[:, :, y1:y2, x1:x2]
    
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Limited to the effective range

    # Reclassify using adversarial examples
    output_perturbed = resnet(perturbed_image)
    final_pred = output_perturbed.max(1, keepdim=True)[1]

    return final_pred, perturbed_image

def resize_sticker(sticker, resize_factor=0.5):
    """ Adjust the sticker size, resize_factor is the scaling factor [0, 1] """
    width, height = sticker.size
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    return sticker.resize((new_width, new_height), Image.ANTIALIAS)

def perturb_image(xs, backimg, sticker, opstickercv, magnification, zstore, searchspace, facemask, transparency=0.5, noise_factor=0.1):
    xs = np.array(xs)
    d = xs.ndim
    if d == 1:
        xs = np.array([xs])
    w, h = backimg.size
    
    # Resize stickers
    sticker = resize_sticker(sticker, resize_factor=0.3)  # Reduce the sticker size, resize_factor can be adjusted according to needs

    # Add global noise to the background image
    backimg_noisy = add_global_noise(backimg, noise_factor)  # Adding global noise
    
    # Apply bilateral filtering to reduce noise
    backimg_noisy = bilateral_filter(backimg_noisy)  # Apply bilateral filtering to a noisy image

    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        rt_sticker = rotate.rotate_bound_white_bg(opstickercv, angle)
        nsticker, _ = mapping3d.deformation3d(sticker, rt_sticker, magnification, zstore, x, y)
        
        # Adjust transparency to create transparent stickers
        sticker_with_alpha = nsticker.convert("RGBA")
        alpha = sticker_with_alpha.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(transparency)  # Adjust transparency
        sticker_with_alpha.putalpha(alpha)

        # Apply transparent stickers to noisy images
        outImage = stick.make_stick2(backimg=backimg_noisy, sticker=sticker_with_alpha, x=x, y=y, factor=xs[i][1])

        # Application semantic attacks
        outImage = semantic_attack(outImage, brightness_factor=1.1, contrast_factor=1.2, color_factor=1.01)

        imgs.append(outImage)

        check_result = int(check_valid(w, h, nsticker, x, y, facemask))
        valid.append(check_result)
        
    return imgs, valid


def check_valid(w, h, sticker, x, y, facemask):
    _, basemap = stick.make_basemap(width=w, height=h, sticker=sticker, x=x, y=y)
    area = np.sum(basemap)
    overlap = facemask * basemap
    retain = np.sum(overlap)
    if abs(area - retain) > 15:
        return 0
    else:
        return 1

def simple_perturb(xs, backimg, sticker, searchspace, facemask, transparency=0.8):
    xs = np.array(xs)
    d = xs.ndim
    if d == 1:
        xs = np.array([xs])
    w, h = backimg.size
    
    sticker = resize_sticker(sticker, resize_factor=0.3)  # Reduce sticker size
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        print(f'making {i}-th perturbed image', end='\r')
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        stickercv = rotate.img_to_cv(sticker)
        rt_sticker = rotate.rotate_bound_white_bg(stickercv, angle)

        # Adjust transparency to create transparent stickers
        sticker_with_alpha = rt_sticker.convert("RGBA")
        alpha = sticker_with_alpha.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(transparency) 
        sticker_with_alpha.putalpha(alpha)

        # Apply transparent stickers to background images
        outImage = stick.make_stick2(backimg=backimg, sticker=sticker_with_alpha, x=x, y=y, factor=xs[i][1])
        imgs.append(outImage)

        check_result = int(check_valid(w, h, rt_sticker, x, y, facemask))
        valid.append(check_result)

    return imgs, valid

def predict_type_facenet(image_perturbed, cleancrop):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def collate_fn(x):
        return x

    loader = DataLoader(
        image_perturbed,
        batch_size=42,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = torch.load('/home/featurize/work/common/resnet_model.pth', map_location='cuda:0').to(device)
    resnet.eval()
    resnet.classify = True
    
    percent = []
    typess = []
    
    for X in loader:
        C = mtcnn(X)
        C = [cleancrop if x is None else x for x in C]
        batch_t = torch.stack(C)
        batch_t = batch_t.to(device)
        out = resnet(batch_t).cpu()
        with torch.no_grad():
            _, indices = torch.sort(out.detach(), descending=True)
            percentage = torch.nn.functional.softmax(out.detach(), dim=1) * 100
            
            for i in range(len(out)):
                cla = [indices[i][0].item(), indices[i][1].item(), indices[i][2].item(), \
                       indices[i][3].item(), indices[i][4].item()]
                typess.append(cla)
                tage = percentage[i]
                percent.append(tage)

    return typess, percent

def initial_predict_facenet(image_perturbed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    resnet = torch.load('/home/featurize/work/common/resnet_model.pth', map_location='cuda:0').to(device)
    resnet.eval()
    resnet.classify = True
    
    percent = []
    typess = []

    C = mtcnn(image_perturbed, save_path='./test.jpg')   # return tensor list
    batch_t = torch.stack(C)
    batch_t = batch_t.to(device)
    out = resnet(batch_t).cpu()
    with torch.no_grad():
        _, indices = torch.sort(out.detach(), descending=True)
        percentage = torch.nn.functional.softmax(out.detach(), dim=1) * 100
        cla = [indices[0][0].item(), indices[0][1].item(), indices[0][2].item(), \
               indices[0][3].item(), indices[0][4].item()]
        typess.append(cla)
        tage = percentage[0]
        percent.append(tage)

    return typess, percent, C[0]
