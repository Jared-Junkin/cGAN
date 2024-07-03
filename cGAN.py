import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torchvision.utils import make_grid
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import os
# copying my download function exactly from the encoder-decoder project
# only difference is that now we won't want to flatten the images

def download_and_process(batch_size:int=16) -> Tuple[DataLoader, Dict[int, List[torch.Tensor]]]:
    """
    Downloads the MNIST dataset and processes it to separate even and odd digits,
    normalizes images to be between -1 and 1, and pads images to be 32x32.
    
    Parameters:
    - batch_size: Number of samples per batch to load.
    
    Returns:
    - DataLoader object for even digits.
    - Dictionary object for odd digits, with digits as keys and image tensors as values.
    """
    # Define a transform to normalize the data, convert to tensors, pad images, and no flattening
    transform = transforms.Compose([
        transforms.Pad(2),  # Center pad the images to increase from 28x28 to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalizes tensors to be between -1 and 1
    ])
    
    # Download the training and test datasets with transformations applied
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # make dictionary object from odd dataset (where dict['1'] = torch.tensor(all 1's))
    odd_dict = {1: [], 3: [], 5:[], 7:[], 9:[]}
    even = []
    for i, (image, label) in enumerate(dataset):
        # even
        if label % 2 == 0:
            even.append(i)
        # odd
        else:
            odd_dict[label].append(image)


    # Create subset and DataLoader for even digits
    even_dataset = Subset(dataset, even)
    even_loader = DataLoader(even_dataset, batch_size=batch_size, shuffle=True)
    return even_loader, odd_dict


def print_model_summary(model, input_size, device):
    """
    Print the output shape of each layer in the model given a specific input size.

    Parameters:
    - model: The PyTorch model (instance of nn.Module or nn.Sequential).
    - input_size: The size of the input tensor (excluding batch size), e.g., (1, 28, 28) for MNIST.
    """
    def register_hook(module):
        def hook(module, input, output):
            print(f"{module.__class__.__name__:>20} : {str(output.shape)}")
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Register hook for each layer
    hooks = []
    model.apply(register_hook)

    # Create a dummy input tensor with the specified size and a batch size of 1
    # and forward it through the model to trigger the hooks
    with torch.no_grad():
        x = torch.randn((1, *input_size)).to(device=device)
        model(x)

    # Remove hooks after printing
    for hook in hooks:
        hook.remove()


# I have opted not to do skip connections. skip connections are useful for preserving low-level
# features across the layers, and since we're doing digit translation we don't need that. 
'''
I'm assuming that because we have 5 possible input digits we might translate from, there should
be at least 5 filters in our bottleneck layer. but that of course doesn't make sense in the context of CNN's, 
it makes sense in the context of MLP's


'''
# ignore the dimensions they aren't right.
generator = nn.Sequential(
    # 28 x 28 x 1
    nn.Conv2d(in_channels=1, kernel_size=3, out_channels=32, stride=1, padding='same'),
    # 28 x 28 x 32
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    # 14 x 14 x 32

    nn.Conv2d(in_channels=32, kernel_size=3, out_channels=64, stride=1, padding='same'),
    # 14 x 14 x 64
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    # 7 x 7 x 64

    nn.Conv2d(in_channels=64, kernel_size=3, out_channels=128, stride=1, padding='same'),
    # 7 x 7 x 128
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, kernel_size=3, out_channels=128, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, kernel_size=3, out_channels=128, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    # 3 x 3 x 128

    
    nn.Conv2d(in_channels=128, kernel_size=3, out_channels=256, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    nn.Conv2d(in_channels=256, kernel_size=3, out_channels=256, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    # 3 x 3 x 256

    # I still don't know if I need to add batchnorm/relu to this transposeconv2d layer
    nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=2),
    # 7 x 7 x 128
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    # 14 x 14 x 64
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    # 28 x 28 x 32
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding='same'),
    nn.Tanh() # important! I think this means you'll have to normalize all data going throug this so its -1 to 1 (at least you'll need to normalize what you feed into the discriminator)
)

discriminator = torch.nn.Sequential(
    # 1 x 32 x 32

    nn.Conv2d(in_channels=2, kernel_size=4, out_channels=32, stride=2, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    # 32 x 32 x 16

    nn.Conv2d(in_channels=32, kernel_size=4, out_channels=64, stride=2, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    # 64 x 8 x 8

    nn.Conv2d(in_channels=64, kernel_size=4, out_channels=128, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    # 128 x 8 x 8

    nn.Conv2d(in_channels=128, kernel_size=4, out_channels=1, stride=1, padding=1),
    # nn.Sigmoid(), # don't need, because we're going to use BCEWithLogitsLoss like the paper does.
    # 1 x 8 x 8 
    # important! we need to add mean here!
    nn.AdaptiveAvgPool2d(1)
)
# print_model_summary(generator, input_size=(1, 32,32)) # this prints the dimensions of all the sizes in my neural net
# print("#######################")
# print_model_summary(discriminator, input_size=(1, 32,32)) # this prints the dimensions of all the sizes in my neural net

# main

def sample_input_images(dataset, device, sample_digits=[0, 2, 4, 6, 8]):
    sampled_images = []
    # We create a dictionary to hold our sampled images for each digit
    digit_images = {digit: [] for digit in sample_digits}

    # Loop through the dataset to get at least one of each digit
    for images, labels in dataset:
        for digit in sample_digits:
            # Check if we still need to sample for this digit
            if len(digit_images[digit]) == 0:
                # Find all indices of the current digit in the batch
                indices = (labels == digit).nonzero(as_tuple=True)[0]
                if len(indices) > 0:
                    # Pick the first occurrence of the digit
                    digit_images[digit].append(images[indices[0]].to(device))

        # Break the loop if we have found all digits
        if all(len(digit_images[digit]) > 0 for digit in sample_digits):
            break

    # Collect one sample for each digit
    for digit in sample_digits:
        sampled_images.append(digit_images[digit][0])

    return torch.stack(sampled_images)  # Stack all sampled images into a single tensor


def addNoise(images: torch.Tensor, noise_amnt=0.025) -> torch.Tensor:
    d = images.device
    noise = torch.rand_like(images) * noise_amnt
    noise.to(device=d)
    images += noise
    images = torch.clamp(images, min=-1, max=1)
    return images

def mapping(labels: torch.Tensor, images: torch.Tensor, distributions: Dict[int, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    mapdict = {0: 1, 2: 3, 4: 5, 6: 7, 8: 9}
    output_labels = torch.zeros_like(labels)
    outputs = torch.zeros_like(images)

    # First loop: Update the labels based on the mapping
    for k, v in mapdict.items():
        output_labels[labels == k] = v

    # Second loop: For each label, sample randomly from distributions[label]
    for i, label in enumerate(output_labels):
        # Sample a random image for the current label
        label = label.item()
        if label in distributions:  # it better be
            sampled_idx = torch.randint(0, len(distributions[label]), (1,)).item()  # Random index
            sampled_image = distributions[label][sampled_idx]  # Get the image
            outputs[i] = sampled_image  # Assign the image to the outputs tensor
        else:
            print("wow something is really wrong")

    return output_labels, outputs

def train(dataset: DataLoader, distributions:Dict[int, List[torch.Tensor]], device:str, epochs:int=90, loss_function = torch.nn.BCEWithLogitsLoss(), lr:int=1e-3) -> None:
    gopt = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    dopt = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_losses, g_losses = [], []  # Lists to store losses
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        d_loss_accum, g_loss_accum = 0.0, 0.0
        n_batches = 0
        print(f"starting epoch {epoch}")
        for images, labels in dataset:
            n_batches += 1
            # send images to cuda
            images = images.to(device=device)

            # add noise so our generator isn't deterministic
            noisy_images = addNoise(images).to(device)

            # send images + noise through generator
            generated_images = generator(noisy_images)

            # get desired output labels based on input labels
            output_labels, sampled_outputs=mapping(labels=labels, images=images, distributions=distributions)


            # concatenate with real inputs
            real_inputs = torch.cat((images, sampled_outputs), dim=1).to(device)
            fake_inputs = torch.cat((images, generated_images.detach()), dim=1).to(device)

            # feed both real and fake inputs through discriminator
            d_guesses_real = discriminator(real_inputs).squeeze()
            d_guesses_fake = discriminator(fake_inputs).squeeze()
            


            # calculate the loss for the discriminator and backprop
            loss = loss_function(d_guesses_real, torch.ones_like(output_labels, dtype=torch.float).to(device)) + loss_function(d_guesses_fake, torch.zeros_like(output_labels, dtype=torch.float).to(device))
            dopt.zero_grad()
            loss.backward()
            dopt.step()
            
            # calculate loss for generator and backprop
            # discriminator.eval() # set discriminator to eval mode while training generator(don't do this...)
            generated_images = generator(noisy_images)
            fake_inputs = torch.cat((images, generated_images), dim=1).to(device)
            d_guesses_fake_gen = discriminator(fake_inputs).squeeze()
            loss_gen = loss_function(d_guesses_fake_gen, torch.ones_like(output_labels, dtype=torch.float).to(device))
            gopt.zero_grad()
            loss_gen.backward()
            gopt.step()

            ## this is just for debugging
            # with torch.no_grad():
            #     generated_images = generator(noisy_images)
            #     fake_inputs = torch.cat((images, generated_images), dim=1).to(device)
            #     d_guesses_fake_gen_new = discriminator(fake_inputs).squeeze()
            #     loss_gen_new = loss_function(d_guesses_fake_gen_new, torch.ones_like(output_labels, dtype=torch.float).to(device))


            # plotting
            # loss_gen = loss_gen.cpu()
            # loss = loss.cpu()
            d_loss_accum += loss.item() 
            g_loss_accum += loss_gen.item()
            # Every 10 epochs, plot and save the current loss graph and example images

        print(f"done epoch {epoch}. final batch D loss = {loss}, G loss = {loss_gen}, new G loss = {loss_gen_new}")
        # print(f"D logits for G loss: {d_guesses_fake_gen}. \n {loss_function(d_guesses_fake_gen_new, torch.ones_like(output_labels, dtype=torch.float).to(device))}")
            # discriminator.train() # set back to train mode 
        print(f"###############################")

        d_losses.append(d_loss_accum / n_batches)
        g_losses.append(g_loss_accum / n_batches)
        if epoch % 10 == 0:
            sampled_images = sample_input_images(dataset, device).to(device)
            noisy_images = addNoise(sampled_images).to(device)
            generated_images = generator(noisy_images).detach()

            fig, axs = plt.subplots(5, 2, figsize=(10, 10)) 

            for i in range(5):
                axs[i, 0].imshow(sampled_images[i].cpu().numpy().squeeze(), cmap='gray')
                axs[i, 0].axis('off')
                axs[i, 1].imshow(generated_images[i].cpu().numpy().squeeze(), cmap='gray')
                axs[i, 1].axis('off')

            axs[0, 0].set_title("Input Digit", pad=20)
            axs[0, 1].set_title("Generated Digit", pad=20)

            plt.tight_layout()
            plt.savefig(f'./saves_offline/digit_comparison_epoch_{epoch}.png')
            plt.close()

            torch.save(generator.state_dict(), f'./saves_offline/generator_epoch_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'./saves_offline/discriminator_epoch_{epoch}.pth')

            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(d_losses, label="Discriminator Loss")
            plt.plot(g_losses, label="Generator Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f'./saves_offline/losses_epoch_{epoch}.png')
            plt.close()

            print(f"Saved plots and model weights for epoch {epoch}.")

            # we need ground truth image (we have that, images). then we need a randomly sampled digit from the appropriate transformation



'''
more common to train discriminator first (just because that's what's worked out better for researchers probably.)
important! note that we MUST feed the generators outputs through the discriminator twice,because the gradients flow through the computational graph:
input -> generator -> discriminator -> loss function 
when we train the discriminator. if we don't pass that through again, the tensors remember the discriminator was a part of their process and the gradients won't 
imppact the generator the way they should. 
.detatch() does this: a->b->c->d->e->f. if we call d.detatch the gradients still flow through e->f but not a,b,c or d
pseudocode:
loss_function = torch.nn.BCEWithLogitsLoss() 
for batch in epoch: (batch=(x,y))
    generator_outputs_batch = generator.feed_forward(x + noise)
    d_guesses_real = discriminator.feed_forward(y)
    d_guesses_fake = discriminator.feed_forward(generator_outputs_batch).detatch()
    discriminator_loss = loss_function(d_guesses_real, [1 1 1 1 1 1 1]) + loss_function(d_guesses_fake, [0 0 0 0 0 0 0])
    loss.backward, etc....

    d_guesses_fake_gen = discriminator.feed_forward(generator_outputs_batch)
    g_loss = loss_function(d_guesses_fake_gen, [1 1 1 1 1 1 1])
    loss.backward, etc....
'''

loss_function = torch.nn.BCEWithLogitsLoss() # discriminator loss
# now how do I get the generator loss from this, and how do i hook it all up into a training loop

if __name__ == "__main__":
    os.makedirs('./saves_offline', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 2e-4 # just setting it for now, will change later whne I can't train this successfully
    generator.to(device)
    discriminator.to(device)

    # now we need to pad all images to be 32 x 32
    # now we need to normalize all image to be between -1 and 1
    batch_size = 16
    even_inputs, odd_dict = download_and_process(batch_size=batch_size)
    train(dataset=even_inputs, distributions=odd_dict, device=device, lr = learning_rate)
    # print_model_summary(discriminator, input_size=(2, 32,32), device = device) # this prints the dimensions of all the sizes in my neural net

    ### this code demonstrates that we are normalizing successfully 0to1 and we are padding to 32x32
    # first_batch_images, first_batch_labels = next(iter(train_loader))
    # first_image = first_batch_images[0]
    # print(first_image.size())
    # print(np.array2string(first_image.squeeze().numpy(), threshold=np.inf))


    