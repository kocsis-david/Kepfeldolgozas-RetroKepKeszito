import cv2
import numpy as np

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust the brightness and contrast of an image. The brightness and contrast values are in the range of -255 to 255.
    The formula used is: image = image * contrast + brightness
    :param image: Input image
    :param brightness: Brightness adjustment value (-255 to 255)
    :param contrast: Contrast adjustment value (-255 to 255)
    :return: Adjusted image
    """
    brightness = int((brightness / 100) * 255)
    contrast = int((contrast / 100) * 255)
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def add_yellow_tint(image, intensity=0.3):
    """
    Add a yellow tint to the image. The tint is achieved by blending the image with a yellow color.
    :param image: Input image
    :param intensity: Intensity of yellow tint (0 to 1)
    :return: Tinted image
    """
    yellow = np.full(image.shape, (0, 255, 255), dtype=np.uint8)
    return cv2.addWeighted(image, 1 - intensity, yellow, intensity, 0)

def apply_bw_effect(image, strength=0.5):
    """
    Apply a partial black and white effect. The effect is achieved by blending the image with its grayscale version.
    :param image: Input image
    :param strength: Strength of B&W effect (0 to 1)
    :return: Image with B&W effect
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 1 - strength, gray, strength, 0)

def apply_dark_yellow_tone(image):
    """
    Apply a dark yellow tone to the image. The tone is achieved by blending the image with a dark yellow color. 
    The formula used is: image = image * 0.5 + dark_yellow * 0.5
    :param image: Input image
    :return: Image with dark yellow tone
    """
    dark_yellow = np.full(image.shape, (0, 200, 200), dtype=np.uint8)
    return cv2.addWeighted(image, 0.5, dark_yellow, 0.5, 0)

def apply_paper_texture(image, texture_path, intensity=0.3):
    """
    Apply a paper texture to the image. The texture is blended with the image using the input intensity value. 
    The formula used is: image = image * (1 - intensity) + texture * intensity
    :param image: Input image
    :param texture_path: Path to the paper texture image
    :return: Image with paper texture
    """
    texture = cv2.imread(texture_path)
    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1-intensity, texture, intensity, 0)

def apply_vignette(image, amount=0.5):
    """
    Apply a vignette effect to the image. The effect darkens the corners of the image. The strength of the effect is proportional to the input value. 
    The effect is applied using a Gaussian kernel. 
    The formula used is: image = image * mask + vignette * (1 - mask)
    :param image: Input image
    :param amount: Strength of vignette effect (0 to 1)
    :return: Image with vignette effect
    """
    height, width = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(width, width/2)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height/2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    vignette = np.copy(image)
    for i in range(3):
        vignette[:,:,i] = vignette[:,:,i] * mask
    return cv2.addWeighted(image, 1-amount, vignette, amount, 0)

def apply_noise(image, intensity=20):
    """
    Apply noise to the image. The noise is generated using a Gaussian distribution. The intensity of the noise is proportional to the input value. 
    The noise is added to the image using the formula: image = image + noise * intensity (clipped to 0-255).
    :param image: Input image
    :param intensity: Intensity of noise
    :return: Image with noise
    """
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch) * intensity
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_white_noise(image, intensity=0.1):
    """
    Add white dots to the image. The number of dots is proportional to the intensity. The dots are added randomly to the image.
    :param image: Input image
    :param intensity: Intensity of white noise (0 to 1)
    :return: Image with white noise
    """
    
    for _ in range(int(image.shape[0] * intensity)):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        image[y, x] = (255, 255, 255)

    return image


def add_age_spots(image, num_spots=20):
    """
    Add age spots to the image. Age spots are blurred circles of random radius between 5 and 20 pixels.
    The spots are added to a separate layer and then blended with the original image.
    :param image: Input image
    :param num_spots: Number of age spots to add
    :return: Image with age spots
    """
    spots = np.zeros(image.shape[:2], dtype=np.uint8)
    for _ in range(num_spots):
        x = np.random.randint(0, image.shape[1])
        y = np.random.randint(0, image.shape[0])
        radius = np.random.randint(5, 20)
        cv2.circle(spots, (x, y), radius, (1), -1)
    
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return np.where(spots[:,:,None] == 1, blurred, image)

def apply_frame(image, frame_path):
    """
    Apply a frame to the image. The frame image should have a transparent center. The input image is placed in the center.

    :param image: Input image (numpy array in BGR format)
    :param frame_path: Path to the frame image
    :return: Framed image
    """
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    
    frame=cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    image=cv2.resize(image, (frame.shape[1], frame.shape[0]))
    frame=np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if frame is None:
        print("Error: Frame image not found.")
        exit()
    new_image=np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    centerval=frame[frame.shape[0]//2, frame.shape[1]//2]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if frame[i, j,0]/255 <0.6 or frame[i, j,1]/255<0.6 or frame[i, j,2]/255<0.6:
                new_image[i, j] = frame[i, j]
            else:
                new_image[i, j] = image[i, j]
    
    
    
    # Convert back to OpenCV format (BGR)
    return cv2.cvtColor(np.array(new_image), cv2.COLOR_RGBA2BGR)

def apply_vintage_effect(image_path, output_path, texture_path, frame_path):
    """
    Apply vintage effect to an image.
    :param image_path: Path to input image
    :param output_path: Path to save output image
    :param texture_path: Path to paper texture image
    :param frame_path: Path to frame image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Apply effects
    img = adjust_brightness_contrast(img, brightness=3, contrast=30)
    #img = add_yellow_tint(img, intensity=0.3)
    img = apply_bw_effect(img, strength=0.2)
    img = apply_dark_yellow_tone(img)
    img = apply_paper_texture(img, texture_path, intensity=0.5)
    
    img = apply_vignette(img, amount=1.8)
    img = apply_noise(img, intensity=1.2)
    img = add_white_noise(img, intensity=0.3)
    img=apply_paper_texture(img, 'paper.jpg', intensity=0.15)
    img = add_age_spots(img, num_spots=30)
    
    img = apply_frame(img, frame_path)
    
    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Vintage effect applied and saved to {output_path}")

# Usage example
apply_vintage_effect('prof.png', 'vintage_prof.jpg', 'paper_texture.jpg', 'frame.jpg')