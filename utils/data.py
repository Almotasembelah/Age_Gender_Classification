from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn.functional as F

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
            self.gender_labels = {class_name: torch.tensor(i).float()  for i, class_name in enumerate(df['gender'].value_counts().index.values)}
            self.gender_classes = list(self.gender_labels.keys())
            self.classes.append(self.gender_classes)
            
        if self.age:
            self.age_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['age'].value_counts().index.values)}
            self.age_classes = list(self.age_labels.keys())
            self.classes.append(self.age_classes)
            
        if self.race:
            self.race_labels = {class_name: torch.tensor(i)  for i, class_name in enumerate(df['race'].value_counts().index.values)}
            self.race_classes = list(self.race_labels.keys())
            self.classes.append(self.race_classes)

        if len(self.classes) == 1:
            self.classes = self.classes[0]
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].squeeze()
        img = Image.open(self.root + item['file']) if self.root is not None else Image.open(item['file'])/255.
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

from torchvision import transforms
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