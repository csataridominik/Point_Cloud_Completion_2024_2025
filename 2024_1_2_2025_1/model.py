
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(init_func)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class MultiViewCompletionNet(BaseNetwork):
    def __init__(self, n_views=4, residual_blocks=4, init_weights=True):  # I have changed residual blcoks from 8 to 4!!!!
        super(MultiViewCompletionNet, self).__init__()

        self.n_views = n_views
        
        self.feature_extractor = models.vgg16(pretrained=True).features[:].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG weights

            
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, kernel_size=7, padding=0),  # Single channel for depth map
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        # Residual blocks
        self.middle = nn.Sequential(*[ResnetBlock(256, 2) for _ in range(residual_blocks)])

        # Merge multi-view features
        self.middle_all = nn.Sequential(
            nn.Conv2d(n_views * 256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0)  # Output channel set to 1 for depth map
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        batch_size, n_views, height, width = x.shape  # Expected shape: [batch_size, n_views, height, width]

        # Encode each view individually
        encoded_views = []
        for i in range(n_views):
            encoded_view = self.encoder(x[:, i:i+1, :, :])  # Encode each view independently, keep channel dimension
            encoded_views.append(self.middle(encoded_view))

        # Concatenate encoded features along the channel dimension
        merged_features = torch.cat(encoded_views, dim=1)  # Should result in [batch_size, n_views * 256, H, W]

        # Process concatenated features through `middle_all`
        mean_features = self.middle_all(merged_features)

        # Decode each view by combining its features with `mean_features`
        outputs = []
        for encoded_view in encoded_views:
            combined_features = torch.cat([encoded_view, mean_features], dim=1)  # Shape: [batch_size, 512, H, W]
            decoded = self.decoder(combined_features)
            outputs.append((torch.tanh(decoded) + 1) / 2)  # Normalize output to [0, 1] range

        return torch.stack(outputs, dim=1).squeeze(2)  # Output shape: [batch_size, n_views, 1, height, width]

class MultiViewCompletionNet2(BaseNetwork):
    def __init__(self, n_views=4, residual_blocks=4, init_weights=True):  # I have changed residual blcoks from 8 to 4!!!!
        super(MultiViewCompletionNet2, self).__init__()

        self.n_views = n_views
        
        self.feature_extractor = models.vgg16(pretrained=True).features[:].eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG weights

            
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0),  # Single channel for depth map
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        # Residual blocks
        self.middle = nn.Sequential(*[ResnetBlock(256, 2) for _ in range(residual_blocks)])

        # Merge multi-view features
        self.middle_all = nn.Sequential(
            nn.Conv2d(n_views * 256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0)  # Output channel set to 1 for depth map
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        # x: [batch_size, n_views, height, width, 3]
        
        batch_size, n_views, height, width, channels = x.shape  # Ensure x is properly shaped
        
        # Permute to make the channel dimension compatible with Conv2D: [batch_size * n_views, 3, height, width]
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # Shape: [batch_size, n_views, 3, height, width]
        x = x.view(batch_size * n_views, channels, height, width)  # Merge batch and n_views dimensions

        # Encode each view individually
        encoded_views = []
        for i in range(n_views):
            # Extract features for each view
            view_features = self.encoder(x[i * batch_size: (i + 1) * batch_size])  # Process each view separately
            encoded_views.append(self.middle(view_features))

        # Concatenate encoded features along the channel dimension
        merged_features = torch.cat(encoded_views, dim=1)  # Shape: [batch_size, n_views * 256, H, W]

        # Process concatenated features through `middle_all`
        mean_features = self.middle_all(merged_features)

        # Decode each view by combining its features with `mean_features`
        outputs = []
        for encoded_view in encoded_views:
            combined_features = torch.cat([encoded_view, mean_features], dim=1)  # Shape: [batch_size, 512, H, W]
            decoded = self.decoder(combined_features)
            outputs.append((torch.tanh(decoded) + 1) / 2)  # Normalize output to [0, 1] range

        # Return output in expected shape: [batch_size, n_views, height, width, 3]
        return torch.stack(outputs, dim=1).permute(0, 1, 3, 4, 2)



