from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn.functional as F
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

class FairFaceDataset(Dataset):
    '''
    This class is used to create a dataset for the FairFace dataset. 
    It can be used to create a dataset for multiple tasks or a single task.
    '''
    def __init__(self, df, transform=None, root='/kaggle/input/fairface/FairFace/', gender=True, age=True, race=True):
        self.df = df
        self.transform = transform
        self.root = root
        
        self.gender = gender
        self.age = age
        self.race = race

        self.classes = []
        if self.gender:
            self.gender_labels = {'Male': torch.tensor(0).float(), 'Female':torch.tensor(1).float()}
            self.gender_classes = list(self.gender_labels.keys())
            self.classes.append(self.gender_classes)
            
        if self.age:
            self.age_labels = {'0-2': torch.tensor(0), 
                              '3-9': torch.tensor(1),
                              '10-19': torch.tensor(2),
                              '20-29': torch.tensor(3),
                              '30-39': torch.tensor(4),
                              '40-49': torch.tensor(5),
                              '50-59': torch.tensor(6),
                              '60-69': torch.tensor(7),
                              'more than 70': torch.tensor(8)}
            self.age_classes = list(self.age_labels.keys())
            self.classes.append(self.age_classes)
            
        if self.race:
            self.race_labels =  {'White': torch.tensor(0), 
                              'Latino_Hispanic': torch.tensor(1),
                              'Black': torch.tensor(2),
                              'East Asian': torch.tensor(3),
                              'Indian': torch.tensor(4),
                              'Southeast Asian': torch.tensor(5),
                              'Middle Eastern': torch.tensor(6)}
            self.race_classes = list(self.race_labels.keys())
            self.classes.append(self.race_classes)

        if len(self.classes) == 1:
            self.classes = self.classes[0]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].squeeze()
        img = Image.open(self.root + item['file'])
        if self.transform:
            img = self.transform(img)

        labels = []
        if self.gender:
            gen_label = self.gender_labels[item['gender']]
            labels.append(gen_label)
        if self.age:
            age_label = self.age_labels[item['age']]
            labels.append(age_label)
        if self.race:
            race_label = self.race_labels[item['race']]
            labels.append(race_label)

        if len(labels) > 1:
            return img, labels
        elif len(labels) == 1:
            return img, *labels
        else:
            raise 


#--------------------------Helper Functions--------------------------

def normalize(dataloader):
    mean = torch.tensor([0, 0, 0], dtype=torch.float)
    std = torch.tensor([0, 0, 0], dtype=torch.float)

    for loader in dataloader:
        img, *rest = loader
        n_samples, n_channel, h, w = img.size()
        value_per_channel = img.reshape(n_samples, n_channel, h*w)
        means = value_per_channel.mean(axis=2)
        stds = value_per_channel.std(axis=2)

        mean += means.mean(axis=0)
        std += stds.mean(axis=0)
    mean /= len(dataloader)
    std /= len(dataloader)
    
    return transforms.Normalize(mean=mean,std= std)


def printNetResults(model, img, gender_mapping, age_mapping, race_mapping, mode=0):
    img = img.unsqueeze(0)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.perf_counter()
    yHat = model(img.to(device))
    end_time = time.perf_counter()
    model.train()
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Forward pass took {elapsed_time_ms:.2f} ms")

    yHat = yHat.squeeze()
    if mode == 0:
        gen_pred = yHat[0]
        age_pred = yHat[1:10]
        race_pred = yHat[10:]
    else:
        gen_pred, age_pred, race_pred = yHat
        
    probabilities_age = F.softmax(age_pred)
    probabilities_race = F.softmax(race_pred)
    probabilities_gender = F.sigmoid(gen_pred)
    pred_labels = []
    class_label = gender_mapping[(gen_pred>0.5).int().item()]
    prob_percent = 100*probabilities_gender if class_label=='Female' else 100*(1-probabilities_gender)
    print( f"Class : {class_label} with probability {prob_percent:.2f}%" )
    print('-'*15)
    for i in range(9):
        class_label = age_mapping[i]
        prob_percent = 100*probabilities_age[i]
        print( f"Class {i+1}: {class_label} with probability {prob_percent:.2f}%" )
    
    print('----------------------')
    for i in range(7):
        class_label = race_mapping[i]
        prob_percent = 100.*probabilities_race[i]
        print( f"Class {i+1}: {class_label} with probability {prob_percent:.2f}%" )    

    print('Predictions:')
    print('gender: ', gender_mapping[(gen_pred>0.5).int().item()], 
         'age: ', age_mapping[probabilities_age.argmax()],
         'race: ', race_mapping[probabilities_race.argmax()])


def visualize(model, batch, max_num_filters=-1, device='cuda'):
    from ipywidgets import interact, Dropdown, IntSlider
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import gc
    import numpy as np
    
    # Visualize feature maps for a specific layer
    @torch.no_grad()
    def visualize_layer(model, batch, device, max_num_filters):
        # Free CUDA memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        model.eval()
        model.to(device)
        batch = batch.to(device)
        inputs = {}
        layer_names = []
    
        # Pass the batch through the model to collect intermediate outputs
        x = batch
        for name, layer in model.named_children():
            if layer._get_name() in torch.nn.modules.linear.__all__:  # Stop at linear layers
                break
            elif layer._get_name() in torch.nn.modules.flatten.__all__:
                break
            try:
                x = layer(x)
                inputs[name] = x.to('cpu')
            except:
                break
            # print(f'Layer name {name}')
            # print(x)
            layer_names.append(name)
        batch = batch.to('cpu')
        del x, batch
        # Dictionary to cache figures
        cached_figures = {}
    
        def wrapper(layer_name, filter_idx, image_idx, mode):
            nonlocal cached_figures  # Access cached figures
    
            if layer_name not in inputs:
                print(f"Invalid layer name: {layer_name}")
                return
    
            img = inputs[layer_name]
    
            if mode == 'Single Image':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Display activation maps for all filters for a single image
                if image_idx >= img.shape[0]:
                    print(f"Invalid image index: {image_idx}. Max index is {img.shape[0] - 1}.")
                    return
    
                cache_key = (layer_name, 'single_image', image_idx)
                if cache_key in cached_figures:
                    fig = cached_figures[cache_key]
                    display(fig)
                else:
                    num_filters = img.shape[1] if max_num_filters==-1 else min(max_num_filters, img.shape[1])
                    
                    num_rows = num_filters // 4 if num_filters % 4 == 0 else (num_filters // 4 + 1)
                    fig: Figure = plt.figure(figsize=(15, 5*num_rows))
                    axes = fig.subplots(num_rows, 4) if num_rows > 1 else [fig.add_subplot(1, num_filters, i + 1) for i in range(num_filters)]
    
                    for i, ax in enumerate(axes.flatten() if isinstance(axes, np.ndarray) else axes):
                        if i < num_filters:
                            ax.imshow(img[image_idx, i].detach().cpu().numpy(), cmap='viridis')
                            ax.axis('off')
                            ax.set_title(f"Filter {i}")
                    fig.tight_layout()
                    cached_figures[cache_key] = fig
                    plt.show()
    
            elif mode == 'Entire Batch':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Display activation maps for a specific filter across the entire batch
                if filter_idx >= img.shape[1]:
                    print(f"Invalid filter index: {filter_idx}. Max index is {img.shape[1] - 1}.")
                    return
    
                cache_key = (layer_name, 'entire_batch', filter_idx)
                if cache_key in cached_figures:
                    fig = cached_figures[cache_key]
                    display(fig)
                else:
                    num_images = img.shape[0]
                    num_rows = num_images // 4 if num_images % 4 == 0 else (num_images // 4 + 1)
                    fig: Figure = plt.figure(figsize=(15, num_rows * 3))
                    axes = fig.subplots(num_rows, 4) if num_rows > 1 else [fig.add_subplot(1, num_images, i + 1) for i in range(num_images)]
    
                    for i, ax in enumerate(axes.flatten() if isinstance(axes, np.ndarray) else axes):
                        if i < num_images:
                            ax.imshow(img[i, filter_idx].detach().cpu().numpy(), cmap='viridis')
                            ax.axis('off')
                            ax.set_title(f"Image {i}")
                    fig.tight_layout()
                    cached_figures[cache_key] = fig
                    plt.show()
    
        return wrapper
    

    wrapper = visualize_layer(model, batch[0], device, max_num_filters)
    
    # Interactive visualization
    layer_names = [name for name, layer in model.named_children()]
    filter_indices = IntSlider(min=0, max=63, step=1, value=0, description="Filter")
    image_indices = IntSlider(min=0, max=batch[0].size(0) - 1, step=1, value=0, description="Image")
    mode_selector = Dropdown(options=['Single Image', 'Entire Batch'], value='Single Image', description="Mode")
    
    interact(wrapper, layer_name=layer_names, filter_idx=filter_indices, image_idx=image_indices, mode=mode_selector)


def display_classified_images(model, loader, dataset, max_images=16, device='cuda'):
    """
    Creates a function to display images with their true and predicted labels.

    This function returns a callable that, when executed, visualizes a batch of images from the dataset along with 
    their predicted and actual labels for gender, age, and race. The labels are color-coded: 
    - Green if the prediction is correct.
    - Red if the prediction is incorrect.

    Parameters:
        - model (torch.nn.Module): The trained PyTorch model used for classification.
        - loader (torch.utils.data.DataLoader): DataLoader providing batches of images and their labels.
        - dataset: The dataset object containing class mappings for gender, age, and race.
        - max_images (int, optional): The maximum number of images to display at a time. Default is 16.
        - device (str, optional): The computation device ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        - function: A callable that, when executed, displays the next batch of images with their classifications.

    Usage Example:
        ```
        viewer = display_classified_images(model, dataloader, dataset)
        viewer()  # Displays the first batch
        viewer()  # Displays the next batch
        ```
    """
    iter_loader = iter(loader)

    def get_classes():
        nonlocal iter_loader
        try:
            imgs, labels = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            imgs, labels = next(iter_loader)

        imgs = imgs.to(device)
        model.eval()
        with torch.no_grad(): 
            preds = model(imgs)
        
        preds = preds.detach().cpu()  
        gen_pred = preds[:, 0]
        age_pred = preds[:, 1:10]
        race_pred = preds[:, 10:]

        age_pred = F.softmax(age_pred, dim=1).argmax(dim=1)
        race_pred = F.softmax(race_pred, dim=1).argmax(dim=1)
        gen_pred = (gen_pred > 0.5).int()

        classes = []
        for i, (g_pred, a_pred, r_pred, g, a, r) in enumerate(zip(gen_pred, age_pred, race_pred,
                                                                 labels[0], labels[1], labels[2])):
            classes.append([imgs[i], g_pred.item(), a_pred.item(), r_pred.item(),
                            g.int().item(), a.item(), r.item()])
        return classes

    def wrapper():
        classes = get_classes()
        n_images = min(len(classes), max_images)
        n_rows = n_images // 4 if n_images % 4 == 0 else n_images // 4 + 1
        fig = plt.figure(figsize=(12, 5 * n_rows))
        axes = fig.subplots(n_rows, 4)

        for i, ax in enumerate(axes.flatten() if isinstance(axes, np.ndarray) else [axes]):
            if i < n_images:
                img, g_pred, a_pred, r_pred, g_true, a_true, r_true = classes[i]

                # Decode labels
                g_pred_text = dataset.classes[0][g_pred]
                g_true_text = dataset.classes[0][g_true]
                a_pred_text = dataset.classes[1][a_pred]
                a_true_text = dataset.classes[1][a_true]
                r_pred_text = dataset.classes[2][r_pred]
                r_true_text = dataset.classes[2][r_true]

                # Determine colors and label formats
                g_label = f"Gen: {g_true_text}" if g_pred == g_true else f"Gen: {g_true_text} (Pred: {g_pred_text})"
                g_color = 'green' if g_pred == g_true else 'red'

                a_label = f"Age: {a_true_text}" if a_pred == a_true else f"Age: {a_true_text} (Pred: {a_pred_text})"
                a_color = 'green' if a_pred == a_true else 'red'

                r_label = f"Race: {r_true_text}" if r_pred == r_true else f"Race: {r_true_text} (Pred: {r_pred_text})"
                r_color = 'green' if r_pred == r_true else 'red'

                # Display the image
                ax.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
                ax.axis('off')

                # Add true labels and incorrect predictions
                ax.text(0.5, -0.2, g_label, fontsize=8, color=g_color, ha='center', va='top', transform=ax.transAxes)
                ax.text(0.5, -0.3, a_label, fontsize=8, color=a_color, ha='center', va='top', transform=ax.transAxes)
                ax.text(0.5, -0.4, r_label, fontsize=8, color=r_color, ha='center', va='top', transform=ax.transAxes)
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    return wrapper

def create_misclassification_viewer(model, dataloader, class_mappings, device='cuda'):
    """
    Creates a function to display misclassified (False Negative) or correctly classified (True Positive) images.

    Parameters:
        - model: Trained PyTorch model
        - dataloader: PyTorch DataLoader
        - class_mappings: Dictionary mapping class indices to class names for age and race
        - device: Computation device ('cuda' or 'cpu')

    Returns:
        - A function that displays 32 misclassified (FN) or correctly classified (TP) images per call.
    """

    misclassified_images = {age: {pred_age: [] for pred_age in class_mappings[1]} for age in class_mappings[1]}
    misclassified_race = {race: {pred_race: [] for pred_race in class_mappings[2]} for race in class_mappings[2]}
    misclassified_images.update(misclassified_race)

    def extract_misclassified_samples():
        """Processes the dataset and stores misclassified samples."""
        for images, labels in dataloader:
            model.eval()
            images = images.to(device)

            with torch.no_grad():
                predictions = model(images)
                age_predictions = F.softmax(predictions[:, 1:10], dim=1).argmax(dim=1)
                race_predictions = F.softmax(predictions[:, 10:], dim=1).argmax(dim=1)

            for img, age_label, race_label, age_pred, race_pred in zip(images, labels[1], labels[2], age_predictions, race_predictions):
                true_age = class_mappings[1][int(age_label.item())]
                predicted_age = class_mappings[1][age_pred.item()]

                true_race = class_mappings[2][int(race_label.item())]
                predicted_race = class_mappings[2][race_pred.item()]

                img_np = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

                if true_age != predicted_age:
                    misclassified_images[true_age][predicted_age].append(img_np)

                if true_race != predicted_race:
                    misclassified_images[true_race][predicted_race].append(img_np)

    extract_misclassified_samples()

    def display_misclassified(true_label, predicted_label):
        """
        Displays misclassified images for a given true and predicted label.

        Parameters:
            - true_label: The actual label
            - predicted_label: The incorrectly predicted label
        """
        batch_size = 32
        images = misclassified_images[true_label][predicted_label]
        index = 0

        def show_next_batch():
            nonlocal index
            if index * batch_size >= len(images):
                print("No more images to display.")
                return

            batch_images = images[index * batch_size: (index + 1) * batch_size]
            index += 1

            num_images = len(batch_images)
            num_cols = min(8, num_images)
            num_rows = (num_images + num_cols - 1) // num_cols  

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))

            if num_rows == 1:
                axes = [axes] if num_cols == 1 else axes.flatten()
            else:
                axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i < num_images:
                    ax.imshow(batch_images[i])
                    ax.axis('off')
                else:
                    ax.set_visible(False)  # Hide empty subplots
            plt.set_title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
            plt.tight_layout()
            plt.show()

        return show_next_batch

    return display_misclassified


def grad_cam(model, image, target_class, target_layer):
    """
    Generate Grad-CAM heatmap for a given model and image.
    Args:
        model: The trained model.
        image: Input image tensor (C, H, W).
        target_class: The class index for which Grad-CAM is computed.
        target_layer: The name of the target convolutional layer.
    """
    model.eval()
    activations = {}
    gradients = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Hook for capturing activations
    def forward_hook(module, input, output):
        activations['value'] = output

    # Hook for capturing gradients
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    # Register hooks
    target_conv_layer = dict(model.model.named_modules())[target_layer]
    target_conv_layer.register_forward_hook(forward_hook)
    target_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    image = image.unsqueeze(0).to(device)
    output = model(image)
    preds = output.detach().cpu()
    gen_pred = preds[:, 0]
    age_pred = preds[:, 1:10]
    race_pred = preds[:, 10:]
    age_pred = F.softmax(age_pred, dim=1)
    race_pred = F.softmax(race_pred, dim=1)
    gen_pred = (gen_pred > 0.5).int()

    if target_class == 1:
        target_class = age_pred.argmax().item() + 1
    elif target_class == 2:
        target_class = 10 + race_pred.argmax().item()
    target_score = output[0, target_class]
    
    # Backward pass
    model.zero_grad()
    target_score.backward(retain_graph=True)

    # Get activations and gradients
    act = activations['value'].detach()
    grad = gradients['value'].detach()
    
    # Global average pooling of gradients
    weights = grad.mean(dim=(2, 3), keepdim=True)

    # Weighted combination of activations
    cam = (weights * act).sum(dim=1).squeeze()

    # Normalize the heatmap
    cam = F.relu(cam)  # ReLU ensures the heatmap is non-negative
    cam -= cam.min()
    cam /= cam.max()

    return cam.cpu().numpy(), gen_pred.item(), age_pred.argmax().item(), race_pred.argmax().item()
    
def upsampleHeatmap(map, img):
    m,M = map.min(), map.max()
    map = 255 * ((map-m) / (M-m))
    map = np.uint8(map)
    map = cv2.resize(map, (img.shape[0], img.shape[1]))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map*0.6 + img*0.4)
    return map

def display_image_and_gradcam(model, image, target_layer, val_dataset):
    """
    Display the original image and Grad-CAM heatmaps for each predicted class.
    Args:
        model: Trained model.
        image: Input image tensor (C, H, W).
        target_layer: Name of the convolutional layer to compute Grad-CAM.
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for display
    image_np = (image_np * 255).astype(np.uint8)
    
    # Generate Grad-CAM heatmaps for each predicted class
    gen_heatmap, gen, age, race = grad_cam(model, image, target_class=0, target_layer=target_layer)  # Gender
    age_heatmap, _, _, _ = grad_cam(model, image, target_class=1, target_layer=target_layer)  # Age
    race_heatmap, _, _, _ = grad_cam(model, image, target_class=2, target_layer=target_layer)  # Race

    # Overlay heatmaps on the original image
    gen_overlay = upsampleHeatmap(gen_heatmap, image_np)
    age_overlay = upsampleHeatmap(age_heatmap, image_np)
    race_overlay = upsampleHeatmap(race_heatmap, image_np)

    # Display the results
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(gen_overlay)
    axes[1].set_title(f"Gender Grad-CAM| pred: {val_dataset.gender_classes[gen]}")
    axes[1].axis("off")

    axes[2].imshow(age_overlay)
    axes[2].set_title(f"Age Grad-CAM| pred: {val_dataset.age_classes[age]}")
    axes[2].axis("off")

    axes[3].imshow(race_overlay)
    axes[3].set_title(f"Race Grad-CAM| pred: {val_dataset.race_classes[race]}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

def display_images_and_gradcam_v1(model, images, target_layer, val_dataset):
    """
    Display the original images and Grad-CAM heatmaps for each predicted class.
    Args:
        model: Trained model.
        images: A batch of input image tensors (N, C, H, W).
        target_layer: Name of the convolutional layer to compute Grad-CAM.
    """
    model.eval()
    batch_size = images.size(0)
    
    # Convert image tensors to numpy for visualization
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to NHWC format
    images_np = (images_np * 255).astype(np.uint8)  # Rescale to [0, 255]

    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes[None, :]  # Ensure axes is always 2D for consistency

    for i in range(batch_size):
        # Generate Grad-CAM heatmaps for each predicted class
        gen_heatmap, gen, age, race = grad_cam(model, images[i], target_class=0, target_layer=target_layer)  # Gender
        age_heatmap, _, _, _ = grad_cam(model, images[i], target_class=1, target_layer=target_layer)  # Age
        race_heatmap, _, _, _ = grad_cam(model, images[i], target_class=2, target_layer=target_layer)  # Race

        # Overlay heatmaps on the original image
        gen_overlay = upsampleHeatmap(gen_heatmap, images_np[i])
        age_overlay = upsampleHeatmap(age_heatmap, images_np[i])
        race_overlay = upsampleHeatmap(race_heatmap, images_np[i])

        # Plot original image and Grad-CAM overlays
        axes[i, 0].imshow(images_np[i])
        axes[i, 0].set_title(f"Original Image {i + 1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gen_overlay)
        axes[i, 1].set_title(f"Gender Grad-CAM {i+1} pred: {val_dataset.gender_classes[gen]}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(age_overlay)
        axes[i, 2].set_title(f"Age Grad-CAM {i+1} pred: {val_dataset.age_classes[age]}")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(race_overlay)
        axes[i, 3].set_title(f"Race Grad-CAM {i+1} pred: {val_dataset.race_classes[race]}")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()
