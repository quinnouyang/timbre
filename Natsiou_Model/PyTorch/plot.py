import matplotlib.pyplot as plt
import torch

from Natsiou_Model.PyTorch.train import VAE
from helper_data import get_dataloaders_mnist
from helper_plotting import plot_generated_images
from helper_plotting import plot_latent_space_with_labels
from helper_plotting import plot_images_sampled_from_vae

if __name__ == '__main__':
    # Hyperparameters
    RANDOM_SEED = 123
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 256
    NUM_EPOCHS = 50
    # Device
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)

    log_dict = torch.load("tensor.pt")

    model = VAE()
    model.load_state_dict(torch.load("tensor.pt"))

    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
        batch_size=BATCH_SIZE,
        num_workers=2,
        validation_fraction=0., )

    plot_generated_images(data_loader=train_loader, model=model, device=DEVICE, modeltype='VAE')

    plot_latent_space_with_labels(
        num_classes=2,
        data_loader=train_loader,
        encoding_fn=model.encoding_fn,
        device=DEVICE)

    plt.legend()
    plt.show()

    with torch.no_grad():
        new_image = model.decoder(torch.tensor([-0.0, 0.03]).to(DEVICE))
        new_image.squeeze_(0)
        new_image.squeeze_(0)
    plt.imshow(new_image.to('cpu').numpy(), cmap='binary')
    plt.show()

    for i in range(1):
        plot_images_sampled_from_vae(model=model, device=DEVICE, latent_size=2)
        plt.show()
