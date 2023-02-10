from torch import nn

class VinyalsConvLayer(nn.Module):
    def __init__(self, in_dim = 32, out_channels = 32, track_running_stats=True) -> None:
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=(3, 3)),
            nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
    def forward(self, x):
        return self.main(x)

class VinyalsConv(nn.Module):
    def __init__(self, num_classes, num_layers = 4, embedding_feats = 28800, track_running_stats=True) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                VinyalsConvLayer(3, track_running_stats=track_running_stats)
            ]
        )
        
        for _ in range(1, num_layers):
            self.layers.append(VinyalsConvLayer(track_running_stats=track_running_stats))
            
        self.flatten = nn.Flatten()
        
        self.ff = nn.Linear(embedding_feats, num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.flatten(x)
        
        return self.ff(x)