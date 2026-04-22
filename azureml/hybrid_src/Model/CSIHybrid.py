import torch 
import torch.nn as nn

class CSIHybrid(nn.Module):
    """
    A hybrid model that combines a ResNet backbone with a Transformer encoder for CSI data classification.
    """
    def __init__(self, num_classes=7, time_steps=250):
        super(CSIHybrid, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3)) 
        )

        self.d_model = 32 * 15

        self.pos_encoder = nn.Parameter(torch.randn(1, time_steps, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=1024, 
            dropout=0.1,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous() 
        x = x.view(B, T, C * F)                 
        
        x = x + self.pos_encoder
        
        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        
        return self.classifier(x)