For this project, we use Analog AI Accelerator API from IBM to overload a customized GAN model for image generation.

For the GAN architecture, we use one ResNet18 as the generator and a ResNet18 as discriminator, they share the standard ResNet Architecture except the output dimemsion.

For the data set, we use MINST data set with 60000 images, we only use the images without labels.

For the training, we use aihwkit package to overload the architecture of ResNet 18 and use hardware-aware training to train our model.

For the main part, we fine-tune the w_noise and out_noise parameters for better hardware-aware model with W&B.

In all, we find that w_noise has huge impact on the model loss compared with out_noise.

Some experiment is below:

![Example Image](./results/GAN_USE/replay_fake_images_gan.gif)

To run the experiment, you need to follow the instructions on this page to compile aihwkit with CUDA:

https://aihwkit.readthedocs.io/en/latest/advanced_install.html

Try In-place installation and install all the dependencies.

For the python files, test.py is just a file to verify your intallation of aihwkit. train_demo.py is the main model and training file. train_wandb.py is 
a file to fine tune the combination of out_noise, w_noise and DAC, ADC. w_noise.py and out_noise.py are files for tuning with w_noise and out_noise each
separately. All the images generated are store in results. disc.pth and gen.pth are final models.

For each files, there is no any commend line parameters.

For all the codes, we refer to the aihwkit examples for our own models.
