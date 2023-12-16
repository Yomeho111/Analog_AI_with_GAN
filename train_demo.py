'''
This script is just for using MNIST data set to train a simple GAN model with standard Pytorch framework
and see the perfomance for this frame. In this part, we use distributed training with 4 GPUS
'''


import os
import glob
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST

from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import matplotlib.image as plt_image
from matplotlib import animation

# import torch.backends.cudnn as cudnn

#import wandb
#wandb.login()

from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.rpu_base import cuda


from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation

# import argparse

# Using inference/hardware-aware training tile
rpu_config = InferenceRPUConfig()

# specify additional options of the non-idealities in forward to your liking
rpu_config.forward.inp_res = 1/64.  # 6-bit DAC discretization.
rpu_config.forward.out_res = 1/256. # 8-bit ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.02   # Some short-term w-noise.
rpu_config.forward.out_noise = 0.02 # Some output noise.

# specify the noise model to be used for inference only
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0) # the model described

# specify the drift compensation
rpu_config.drift_compensation = GlobalDriftCompensation()


# Set your parameters
SEED = 1
N_EPOCHS = 50
Z_DIM = 28
DISPLAY_STEP = 20
BATCH_SIZE = 512
LR = 2e-2
NUM_WORKERS = 4
INPUT = 28 * 28 * 1

# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda:1" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "GAN_USE")



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = AnalogConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, rpu_config = rpu_config)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AnalogConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, rpu_config = rpu_config)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = AnalogSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            
            self.shortcut = AnalogSequential(
                AnalogConv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False, rpu_config = rpu_config),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.out1 = AnalogSequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.conv2,
            self.bn2,
        )
        self.out2 = AnalogSequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.out1(x)
        out += self.shortcut(x)
        out = self.out2(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_out_put):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = AnalogConv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, rpu_config = rpu_config)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = AnalogLinear(512*block.expansion, num_out_put, rpu_config = rpu_config)

        self.out = AnalogSequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            self.linear,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return AnalogSequential(*layers)

    def forward(self, x):
        return self.out(x)


def ResNet18(output):
    return ResNet(BasicBlock, [2, 2, 2, 2], output)


def store_tensor_images(image_tensor, label, current_step, num_images=25, size=(1, 28, 28)):
    """Store images using a uniform grid.

    Given a tensor of images, number of images, and size per image, stores the
    images using a uniform grid.

    Args:
        image_tensor (Tensor): tensor of images
        label (str): text label
        current_step (int): current step number
        num_images (int): number of images
        size (Tuple): shape of images
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    save_image(image_grid, os.path.join(RESULTS, f"{label}_step_{current_step}.png"))


def show_animation_fake_images():
    """Display images using a matplotlib animation.

    Displays every image labeled as "fake_images_step_*.png" inside the
    results/GAN folder using a matplotlib animation.
    """
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    sorted_available_images = sorted(
        glob.glob(f"{RESULTS}/fake_*.png"), key=lambda s: int(s.split("_")[-1].split(".png")[0])
    )
    ims = [[plt.imshow(plt_image.imread(i))] for i in sorted_available_images]
    ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    animation_writer = animation.PillowWriter()

    ani.save(os.path.join(RESULTS, "replay_fake_images_gan.gif"), writer=animation_writer)
    plt.show()


class Generator(nn.Module):
    """Generator Class.

    Args:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, output_size = 784):
        super().__init__()
        # Build the neural network.
        self.gen = ResNet18(output_size)

    def forward(self, noise):
        """Complete a forward pass of the generator.

        Given a noise tensor, returns generated images.

        Args:
            noise (Tensor): a noise tensor with dimensions (n_samples, z_dim)

        Returns:
            Tensor: the generated images.
        """
        return self.gen(noise)



def get_noise(n_samples, z_dim, device="cpu"):
    """Create noise vectors.

    Given the dimensions (n_samples, z_dim), creates a tensor of that shape
    filled with random numbers from the normal distribution.

    Args:
        n_samples (int): the number of samples to generate, a scalar
        z_dim (int): the dimension of the noise vector, a scalar
        device (device): the device type

    Returns:
        Tensor: random vector
    """
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device
    # argument to the function you use to generate the noise.
    return torch.randn(n_samples, 1, z_dim, z_dim).to(device)



class Discriminator(nn.Module):
    """Discriminator Class.

    Args:
        im_dim (int): the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim (int): the inner dimension, a scalar
    """

    def __init__(self, output_size = 1):
        super().__init__()
        self.disc = ResNet18(output_size)

    def forward(self, image):
        """Complete a forward pass of the discriminator.

        Given an image tensor, returns a 1-dimension tensor representing fake/real.

        Args:
            image (Tensor): a flattened image tensor with dimension (im_dim)

        Returns:
            Tensor: a 1-dimension tensor representing fake/real
        """
        return self.disc(image)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """Return the loss of the discriminator given inputs.

    Args:

        gen (nn.Module): the generator model, which returns an image
            given z-dimensional noise
        disc (nn.Module): the discriminator model, which returns a
            single-dimensional prediction of real/fake
        criterion (nn.Module): the loss function, which should be used
            to compare the discriminator's predictions to the ground
            truth reality of the images (e.g. fake = 0, real = 1)
        real (Tensor): a batch of real images
        num_images (int): the number of images the generator should
            produce, which is also the length of the real images
        z_dim (int): the dimension of the noise vector, a scalar
        device (device): the device type

    Returns:
        Tensor: a torch scalar loss value for the current batch

    """
    noise_vector = get_noise(num_images, z_dim, device)
    generated_images = gen(noise_vector)
    test_fake_im = disc(generated_images.detach().view(-1, 1, 28, 28))
    test_fake_im_loss = criterion(test_fake_im, torch.zeros_like(test_fake_im))

    test_real_im = disc(real.detach())
    test_real_im_loss = criterion(test_real_im, torch.ones_like(test_real_im))

    disc_loss = (test_real_im_loss + test_fake_im_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """Return the loss of the generator given inputs.

    Args:

        gen (nn.Module): the generator model, which returns an image
            given z-dimensional noise
        disc (nn.Module): the discriminator model, which returns a
            single-dimensional prediction of real/fake
        criterion (nn.Module): the loss function, which should be used
            to compare the discriminator's predictions to the ground
            truth reality of the images (e.g. fake = 0, real = 1)
        num_images (int): the number of images the generator should
            produce, which is also the length of the real images
        z_dim (int): the dimension of the noise vector, a scalar
        device (device): the device type

    Returns:
        Tensor: a torch scalar loss value for the current batch

    """
    noise_vector = get_noise(num_images, z_dim, device)
    generated_images = gen(noise_vector)
    test_fake_im = disc(generated_images.view(-1, 1, 28, 28))
    gen_loss = criterion(test_fake_im, torch.ones_like(test_fake_im))

    return gen_loss


def training_loop(gen, disc, gen_opt, disc_opt, criterion, dataloader, n_epochs, display_step):
    """Training loop.

    Args:
        gen (nn.Module): the generator model
        disc (nn.Module): the discriminator model
        gen_opt (Optimizer): analog model optimizer for the generator
        disc_opt (Optimizer): analog model optimizer for the discriminator
        criterion (nn.Module): criterion to compute loss
        dataloader (DataLoader): Data set to train and evaluate the models
        n_epochs (int): global parameter to define epochs number
        display_step (int): defines the period to display the training progress
    """
    # pylint: disable=too-many-locals
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    for _ in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in dataloader:
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset.
            real = real.to(DEVICE)

            # Update discriminator.
            # Zero out the gradients before backpropagation.
            disc_opt.zero_grad()

            # Calculate discriminator loss.
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, Z_DIM, DEVICE)

            # Update gradients.
            disc_loss.backward()

            # Update optimizer.
            disc_opt.step()

            gen_opt.zero_grad()

            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, Z_DIM, DEVICE)

            gen_loss.backward()

            gen_opt.step()

            # Keep track of the average discriminator loss.
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss.
            mean_generator_loss += gen_loss.item() / display_step

            # Visualization code.
            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"{datetime.now().time().replace(microsecond=0)} --- "
                    f"Step {cur_step}: "
                    f"Generator loss: {mean_generator_loss}, "
                    f"discriminator loss: {mean_discriminator_loss}"
                )
                fake_noise = get_noise(cur_batch_size, Z_DIM, device=DEVICE)
                fake = gen(fake_noise)

                store_tensor_images(fake, "fake_images", cur_step)
                # For the example we will store only the fake images generated
                # store_tensor_images(real, 'real_images', cur_step).

                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(SEED)

    # Load MNIST dataset as tensors.
    dataloader = DataLoader(
        MNIST(PATH_DATASET, train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started GAN Example")

    gen = Generator(INPUT).to(DEVICE)
    gen_opt = AnalogSGD(gen.parameters(), lr=LR)
    gen_opt.regroup_param_groups(gen)

    disc = Discriminator().to(DEVICE)
    disc_opt = AnalogSGD(disc.parameters(), lr=LR)
    disc_opt.regroup_param_groups(disc)

    # if USE_CUDA:
    #     gen = torch.nn.DataParallel(gen)
    #     disc = torch.nn.DataParallel(disc)
    #     cudnn.benchmark = True

    print(rpu_config)
    print(gen)
    print(disc)

    criterion = nn.BCEWithLogitsLoss()

    training_loop(gen, disc, gen_opt, disc_opt, criterion, dataloader, N_EPOCHS, DISPLAY_STEP)
    show_animation_fake_images()

    torch.save({
        'model_state_dict': gen.state_dict(),
        'optimizer_state_dict': gen_opt.state_dict(),
        # include other items if necessary
        }, 'gen.pth')

    torch.save({
        'model_state_dict': disc.state_dict(),
        'optimizer_state_dict': disc_opt.state_dict(),
        # include other items if necessary
        }, 'disc.pth')

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed GAN Example")


if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()
