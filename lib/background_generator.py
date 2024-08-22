import numpy as np
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
import random
from lib.background_loader import BackgroundLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Background_Gen:
    """
Class designed to generate synthetic backgrounds and create composite images by blending backgrounds with provided images using masks. The generated backgrounds can be noise-based, shape-based, or complex patterns, and can include various image transformations such as blurring, noise addition, and lighting adjustments.

Args:
    shape (tuple): The shape of the images and backgrounds (batch size, height, width, channels).
    num_noise_templates (int): The number of noise templates to generate for background creation.
    num_shape_templates (int): The number of shape templates to generate for background creation.
    dataset_dir (list): Optional; directory paths to load additional backgrounds from an external dataset.
    dataset_only (bool): If True, only uses images from the dataset as backgrounds. If False, uses both dataset images and synthetic backgrounds.
"""

    def __init__(self, shape=(1, 2048, 2048, 3), num_noise_templates:int=5, num_shape_templates:int=5, dataset_dir:list=None, dataset_only=False) -> None:
        self.shape = shape
        self.num_noise_templates = num_noise_templates
        self.num_shape_templates = num_shape_templates
        self.noise_templates = [self._generate_noise_template(shape[1:], 300) for _ in range(num_noise_templates)]
        self.shape_templates = [self._generate_shape_template(shape[1:]) for _ in range(num_shape_templates)]
        self.single_colors = [(255, 255, 255), (0, 0, 0), (0, 255, 0)]  # white, black, green
        self.flip_degrees = {1: 90, 2: 180, 3: 270}
        self.dataset_only = dataset_only # Whether to use only dataset images or along artificial ones
        
        if dataset_dir:
            self.bg_loader = DataLoader(BackgroundLoader(dataset_dir), batch_size=self.shape[0], shuffle=True)
        else:
            self.bg_loader = None

    def _generate_noise_template(self, shape, patch_size=100):
        noise = np.zeros(shape, dtype=np.uint8)
        for _ in range(10):
            x = np.random.randint(0, shape[1] - patch_size)
            y = np.random.randint(0, shape[0] - patch_size)
            noise_patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
            noise[y:y+patch_size, x:x+patch_size] = noise_patch
        return noise

    def _generate_shape_template(self, shape):
        background = np.zeros(shape, dtype=np.uint8)
        colors = [tuple(np.random.randint(150, 256, 3).tolist()) for _ in range(10)]
        
        for color in colors:
            if np.random.rand() > 0.5:
                center = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
                axes = (np.random.randint(0, 150), np.random.randint(50, 150))
                angle = np.random.randint(0, 360)
                cv2.ellipse(background, center, axes, angle, 0, 360, color, -1)
            else:
                vertices = np.array([[
                    (np.random.randint(0, shape[1]), np.random.randint(0, shape[0])) for _ in range(np.random.randint(3, 10))
                ]], dtype=np.int32)
                cv2.fillPoly(background, vertices, color)
        
        for _ in range(15):
            pt1 = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            pt2 = (np.random.randint(0, shape[1]), np.random.randint(0, shape[0]))
            color = tuple(np.random.randint(0, 256, 3).tolist())
            thickness = np.random.randint(1, 10)
            cv2.line(background, pt1, pt2, color, thickness)
        
        return background
    
    def apply_gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        # Create a Gaussian kernel
        grid = torch.arange(kernel_size).float().cuda()
        grid -= (kernel_size - 1) / 2.0
        kernel = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d.expand(image.size(1), 1, kernel_size, kernel_size)

        # Apply Gaussian blur
        blurred_image = F.conv2d(image, kernel_2d, padding=kernel_size // 2, groups=image.size(1))
        return blurred_image
    

    def apply_high_pass(self, image, kernel_size=3):
        # Create a simple averaging kernel for low-pass filtering
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size * kernel_size)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)  # Repeat for each channel
        low_pass = F.conv2d(image, kernel, padding=kernel_size//2, groups=image.shape[1])
        high_pass = image - low_pass
        return high_pass


    def apply_motion_blur(self, image, kernel_size=5, direction='horizontal'):
        if direction == 'horizontal':
            kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=image.device)
            kernel[:, :, kernel_size // 2, :] = 1.0 / kernel_size
        else:
            kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=image.device)
            kernel[:, :, :, kernel_size // 2] = 1.0 / kernel_size

        # Ensure kernel is correctly sized for the image's channels
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)
        motion_blur = F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])
        return motion_blur
    
    def apply_averaging_blur(self, image, kernel_size=3):
        # Create an averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size * kernel_size)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)  # Repeat for each channel
        
        # Apply averaging blur
        blurred_image = F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])
        return blurred_image

    def _generate_zigzag_template(self, shape, line_thickness=10):
        background = np.zeros(shape, dtype=np.uint8)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        for i in range(0, shape[1], line_thickness * 2):
            cv2.line(background, (i, 0), (0, i), color, line_thickness)
            cv2.line(background, (shape[1] - i, shape[0]), (shape[1], shape[0] - i), color, line_thickness)
        return background

    def _generate_checkerboard_template(self, shape, square_size=50):
        background = np.zeros(shape, dtype=np.uint8)
        color1 = tuple(np.random.randint(0, 256, 3).tolist())
        color2 = tuple(np.random.randint(0, 256, 3).tolist())
        for y in range(0, shape[0], square_size):
            for x in range(0, shape[1], square_size):
                if (x // square_size) % 2 == (y // square_size) % 2:
                    cv2.rectangle(background, (x, y), (x + square_size, y + square_size), color1, -1)
                else:
                    cv2.rectangle(background, (x, y), (x + square_size, y + square_size), color2, -1)
        return background

    def generate_noise_patches(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        template = self.noise_templates[np.random.randint(self.num_noise_templates)]
        for i in range(shape[0]):  # Iterate over batch
            x_shift = np.random.randint(0, shape[2] - template.shape[1] + 1)
            y_shift = np.random.randint(0, shape[1] - template.shape[0] + 1)
            noise[i] = np.roll(template, (y_shift, x_shift), axis=(0, 1))
        return noise

    def generate_complex_shapes_background(self, shape):
        background = np.zeros(shape, dtype=np.uint8)
        template = self.shape_templates[np.random.randint(self.num_shape_templates)]
        for i in range(shape[0]): 
            x_shift = np.random.randint(0, shape[2] - template.shape[1] + 1)
            y_shift = np.random.randint(0, shape[1] - template.shape[0] + 1)
            background[i] = np.roll(template, (y_shift, x_shift), axis=(0, 1))
        return background

    def generate_gradient_background(self, shape):
        gradient = np.zeros(shape, dtype=np.uint8)
        for i in range(shape[0]):  
            for j in range(shape[1]):
                color = np.array([j * 255 / shape[1]] * 3, dtype=np.uint8)
                gradient[i, j, :] = color
        return gradient

    def blend_images(self, image1, image2, alpha):
        blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
        return blended

    def add_gaussian_noise(self, image, mean=0.0, std=0.1):
        noise = torch.randn(image.size(), device=image.device) * std + mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, -1, 1)

    def adjust_lighting(self, image, factor):
        return torch.clamp(image * factor, -1, 1)

    def generate_complex_background_single(self, idx):
        if random.random() < 0.3:
            color = self.single_colors[np.random.randint(len(self.single_colors))]
            single_color_bg = np.full((self.shape[1], self.shape[2], self.shape[3]), color, dtype=np.uint8)
            return single_color_bg

        # Randomly choose between different patterns
        if np.random.rand() < 0.2:
            return self._generate_zigzag_template(self.shape[1:])
        elif np.random.rand() < 0.3:
            return self._generate_checkerboard_template(self.shape[1:])

        shape = (1, *self.shape[1:])
        noise_bg = self.generate_noise_patches(shape)
        shapes_bg = self.generate_complex_shapes_background(shape)
        gradient_bg = self.generate_gradient_background(shape)

        complex_bg = np.zeros_like(noise_bg)
        temp = self.blend_images(noise_bg[0], shapes_bg[0], 0.5)
        complex_bg[0] = self.blend_images(temp, gradient_bg[0], 0.5)

        return complex_bg[0]

    def generate_background(self):
        if self.bg_loader and (self.dataset_only or random.random() < 0.5):
            # Load backgrounds from dataset
            backgrounds = next(iter(self.bg_loader))
            backgrounds = 2 * backgrounds - 1.0
            return backgrounds
        else:
            # Generate complex backgrounds
            with ThreadPoolExecutor() as executor:
                complex_bgs = list(executor.map(self.generate_complex_background_single, range(self.shape[0])))
            backgrounds = np.stack(complex_bgs, axis=0)
            backgrounds = torch.tensor(backgrounds, dtype=torch.float32)  # Convert to PyTorch tensor

        if len(backgrounds.shape) == 4 and backgrounds.shape[1] != 3:
            backgrounds = backgrounds.permute(0, 3, 1, 2)
        # Normalise
        backgrounds = 2 * (backgrounds / 255.0) - 1.0

        return backgrounds  # Return as PyTorch tensor with shape [B, 3, H, W]

    
    def random_flip(self, images: torch.Tensor, masks: torch.Tensor, degrees: int):
        if degrees == 90:
            images = torch.rot90(images, k=1, dims=[2, 3])
            masks = torch.rot90(masks, k=1, dims=[2, 3])
        elif degrees == 180:
            images = torch.rot90(images, k=2, dims=[2, 3])
            masks = torch.rot90(masks, k=2, dims=[2, 3])
        elif degrees == 270:
            images = torch.rot90(images, k=3, dims=[2, 3])
            masks = torch.rot90(masks, k=3, dims=[2, 3])
        
        return images, masks

    def create_composites(self, images:torch.Tensor, masks:torch.Tensor, backgrounds:torch.Tensor, random_flip:bool=False):
        """
        Create composites by blending images with backgrounds using masks.

        Args:
            images (torch.Tensor): Batch of images with shape [B, 3, 1024, 1024].
            masks (torch.Tensor): Batch of masks with shape [B, 3, 1024, 1024].
            backgrounds (torch.Tensor): Batch of backgrounds with shape [B, 3, 1024, 1024].
            flip (bool): If True, applies a random rotation with a 30% probability. The rotation will be by a multiple of 90 degrees (90, 180, or 270 degrees). If False, no rotation is applied.

        Returns:
            torch.Tensor: Batch of composited images with shape [B, 3, 1024, 1024].
        """
       # Ensure masks are binary (0 or 1)
        masks = (masks > 0.5).float()

        # Blend images with backgrounds using masks
        composites = images * masks + backgrounds * (1 - masks)

        # Apply additional transformations with 30% probability
        if random.random() < 0.3:
            if random.random() < 0.5:
                # Add Gaussian noise
                mean = 0.0
                std = 0.2
                composites = self.add_gaussian_noise(composites, mean, std)
            else:
                # Adjust lighting
                factor = random.uniform(0.3, 2.0)  # Adjust brightness randomly between 50% and 150%
                composites = self.adjust_lighting(composites, factor)
        
        # Apply random flip with a 30% probability
        if random_flip and random.random() < 0.2:
            r = random.randint(1, 3)
            composites, masks = self.random_flip(images=composites, masks=masks, degrees=self.flip_degrees[r])
            return composites, masks
        
        if random.random() > 0.5:
            if random.random() < 0.3:
                composites = self.apply_gaussian_blur(composites, kernel_size=5, sigma=1.0)
            elif random.random() < 0.2:
                composites = self.apply_averaging_blur(composites, kernel_size=3)
            if random.random() < 0.2:
                direction = 'horizontal' if random.random() < 0.5 else 'vertical'
                composites = self.apply_motion_blur(composites, kernel_size=5, direction=direction)

        return composites, masks
    

if __name__ == '__main__':
    import time
    start_time = time.time()
    bg_gen = Background_Gen(shape=(2, 1024, 1024, 3))

    images = torch.randn(2, 3, 1024, 1024)  # Batch of left or right images
    masks = torch.randn(2, 3, 1024, 1024)   # Corresponding masks
    backgrounds = bg_gen.generate_background()  # Generated backgrounds
    print(backgrounds.shape)  # Should output: torch.Size([4, 3, 2048, 2048])

    # Ensure masks are binary (0 or 1)
    masks = (masks > 0.5).float()

    # Create composites
    composites = bg_gen.create_composites(images, masks, backgrounds)
    print(f'composites shape {composites.shape}')  # Should output: torch.Size([2, 3, 1024, 1024])

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")