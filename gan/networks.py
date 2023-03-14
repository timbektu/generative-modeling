import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
        stride=1,
    ):
        super(UpSampleConv2D, self).__init__()
        self.upscale_factor = upscale_factor
        self.pix_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2d = nn.Conv2d(in_channels=input_channels, out_channels= n_filters, kernel_size=kernel_size, stride=stride, padding= padding)


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return output

        x = torch.repeat(1,self.upscale_factor,1,1) #TODO: shouldn't this be elementwise?
        x = self.pix_shuffle(x)
        x = self.conv2d(x)

        print(x.shape) #TODO: verify

        return x

class DownSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0, stride=1
    ):
        super(DownSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        self.downscale_factor = downscale_ratio
        self.pix_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        self.conv2d = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=stride, padding=padding)


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output

        x = self.pix_unshuffle(x)
        #TODO:


class ResBlockUp(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        # TODO 1.1: Setup the network layers
        self.ResBlockUp = nn.Sequential([nn.BatchNorm2d(num_features=input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                         nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                         nn.ReLU(),
                                         UpSampleConv2D(input_channels=input_channels, kernel_size=kernel_size, n_filters=n_filters, padding=1, stride=1)])
        self.upsample_residual = UpSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1, stride=1)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        return self.ResBlockUp(x) + self.upsample_residual(x)


class ResBlockDown(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # TODO 1.1: Setup the network layers
        self.ResBlockDown = nn.Sequential([nn.ReLU(), 
                                           nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1),
                                           nn.ReLU(),
                                           DownSampleConv2D(input_channels=n_filters, kernel_size=kernel_size, n_filters=n_filters)
                                           ]) #TODO: What is down-sample doing really? input shape output shape?

        self.downsample_residual = DownSampleConv2D(input_channels=input_channels, n_filters=n_filters, kernel_size=1, stride=1)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        return self.ResBlockDown(x) + self.downsample_residual(x)


class ResBlock(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # TODO 1.1: Setup the network layers
        self.ResBlock = nn.Sequential([nn.ReLU(),
                                        nn.Conv2d(in_channels= input_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(input_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1)])

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        return self.ResBlock(x) + x


class Generator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        self.starting_image_size = starting_image_size
        self.dense_init = nn.Linear(128, 2048, bias = True)
        self.gen_block = nn.Sequential([ResBlockUp(input_channels=128, kernel_size=3, n_filters=128),
                                        ResBlockUp(input_channels=128, kernel_size=3, n_fitlers=128),
                                        ResBlockUp(input_channels=128, kernel_size=3, n_fitlers=128),
                                        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
                                        nn.Tanh()
                                        ])

    @torch.jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!

        z = self.dense_init(z).reshape((-1, 128, self.starting_image_size, self.starting_image_size))
        z = self.gen_block(z)
        return z

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        z = torch.randn(n_samples, 128) #TODO: device and torch.half AMP stuff?
        z = self.dense_init(z).reshape((-1, 128, self.starting_image_size, self.starting_image_size))
        z = self.gen_block(z)
        return z


class Discriminator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.dscr_block = nn.Sequential([ResBlockDown(input_channels=3, kernel_size=3, n_filters=128),
                                         ResBlockDown(input_channels=128, kernel_size=3, n_filters=128),
                                         ResBlock(input_channels=128, kernel_size=3, n_filters=128),
                                         ResBlock(input_channels=128, kernel_size=3, n_filters=128),
                                         nn.ReLU()])
        self.dense_out = nn.Linear(in_features=128, out_features=1, bias=True)


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.

        x = self.dscr_block(x).sum(dim=[2,3])
        x = self.dense_out(x)
        return x
