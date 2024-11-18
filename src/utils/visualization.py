import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch


def tsne_with_projector(
    model,
    dataset,
    image_key="image",
    batch_size=16,
    n_components=2,
    save=False,
    path="../outputs/plots",
    tensorboard_log_dir="../outputs/tensorboard",
    projector_metadata_path="../outputs/metadata.tsv",
    device="cuda",
):
    """
    Compute t-SNE and visualize embeddings in TensorBoard (Embedding Projector and image logs).
    :param model: the model used to encode the data
    :param dataset: sample of data
    :param image_key: key to extract images from dataset batches
    :param batch_size: batch size for data loading
    :param n_components: dimensionality of the points in the plot (2D / 3D)
    :param save: if True, save the plot as an image
    :param path: path where to save the plot
    :param tensorboard_log_dir: directory to save TensorBoard logs
    :param projector_metadata_path: path to save metadata for the embedding projector
    :param device: device to use for computations ('cuda' or 'cpu')
    """
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    model.eval()

    # Initialize TensorBoard writer
    writer = SummaryWriter(tensorboard_log_dir)

    encoded_images = []
    labels = []  # Collect labels for metadata
    for batch in tqdm(data_loader, desc="Encoding images", leave=False):
        with torch.no_grad():
            images = batch[image_key].to(device)
            labels.extend(
                batch.get("label", torch.zeros(len(images))).cpu().numpy()
            )  # Assuming 'label' exists
            encoded = model.encoder(images)
        encoded_images.extend(encoded.cpu().detach().numpy())

    encoded_images = torch.tensor(encoded_images)
    labels = [str(label) for label in labels]  # Convert labels to strings for metadata

    # Save metadata for projector
    with open(projector_metadata_path, "w") as f:
        f.write("Label\n")  # Header for metadata file
        f.writelines("\n".join(labels))
    print(f"Metadata saved to {projector_metadata_path}")

    # Log embeddings for the TensorBoard projector
    print("Logging embeddings to TensorBoard...")
    writer.add_embedding(
        encoded_images,
        metadata=labels,
        global_step=0,
        tag="t-SNE Embeddings",
    )

    # Perform t-SNE for a local plot (optional)
    if n_components == 2:
        print("Performing t-SNE for local plot...")
        embedded = TSNE(n_components=n_components, verbose=1).fit_transform(
            encoded_images
        )

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=embedded[:, 0],
            y=embedded[:, 1],
            hue=labels,
            palette=sns.color_palette("hls", len(set(labels))),
            legend="full",
        )
        plt.tight_layout()
        plt.axis("off")

        # Save plot as image
        if save:
            plt.savefig(path)
            print(f"Plot saved to {path}")

        # Log plot to TensorBoard
        writer.add_figure("t-SNE Projection", plt.gcf())
        plt.close()

    writer.close()
    print(f"TensorBoard logs saved to {tensorboard_log_dir}")
