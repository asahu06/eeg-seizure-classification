import torch.nn as nn

class SeizureCNN2D(nn.Module):
    def __init__(self, in_channels=18):
        super(SeizureCNN2D, self).__init__()

        # Depthwise Convolution: Setting `groups=in_channels` forces the network 
        # to learn spatial/frequency features independently for each EEG channel 
        # (2 feature maps per channel) before mixing them in later layers.
        self.conv1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels * 2,   
            kernel_size  = (3, 3),
            padding      = (1, 1),
            groups       = in_channels        
        )
        self.bn1   = nn.BatchNorm2d(in_channels * 2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels * 2, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.relu  = nn.ReLU()
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x).squeeze(1)