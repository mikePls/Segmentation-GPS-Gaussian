import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x  

# Double Convolution Block with optional dropout
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(DoubleConv, self).__init__()
        num_groups = out_channels // 8  
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return self.double_conv(x)

# Modified UNet with Spatial Attention
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_prob=0.3):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32, dropout_prob)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(32, 64, dropout_prob)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128, dropout_prob)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256, dropout_prob)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512, dropout_prob)
        
        # Attention blocks added before the upsampling layers
        self.att1 = SpatialAttention(kernel_size=7)
        self.att2 = SpatialAttention(kernel_size=7)
        self.att3 = SpatialAttention(kernel_size=7)
        self.att4 = SpatialAttention(kernel_size=7)
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(512, 256, dropout_prob)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(256, 128, dropout_prob)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(128, 64, dropout_prob)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(64, 32, dropout_prob)
        self.outc1 = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)
        x5 = self.down4(x4)
        x5 = self.conv4(x5)
        
        # Apply attention after downsampling
        x5 = self.att1(x5) * x5
        x4 = self.att2(x4) * x4
        x3 = self.att3(x3) * x3
        x2 = self.att4(x2) * x2
        
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv5(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv6(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv7(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv8(x)
        out1 = self.outc1(x)
        return out1

if __name__ == '__main__':

    # Dummy batch of left and right images and masks
    batch_size = 1
    channels = 3
    height = 1024
    width = 1024

    left_images = torch.rand(batch_size, channels, height, width)
    right_images = torch.rand(batch_size, channels, height, width)
    left_masks = torch.rand(batch_size, channels, height, width)
    right_masks = torch.rand(batch_size, channels, height, width)

    # Concatenate left and right images along the batch dimension
    input_images = torch.cat([left_images, right_images], dim=0).cuda()  # Move to GPU
    print(f'Input shape {input_images.shape}')

    # Initialize the model
    model = UNet(n_channels=3, n_classes=3)  # 3 channels for each image, 3 classes for RGB mask output
    model.cuda()  # Move model to GPU

    # Forward pass
    output_masks = model(input_images)

    # Check the output shape
    print("Output shape of masks:", output_masks.shape)

    # Extract left and right masks
    output_left_masks = output_masks[:batch_size]
    output_right_masks = output_masks[batch_size:]

    print("Output shape of left masks:", output_left_masks.shape)
    print("Output shape of right masks:", output_right_masks.shape)
