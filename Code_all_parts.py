"""
Architecture 1 """


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import pandas as pd
import random
import wandb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
L1_LAMBDA = 100
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


class ISICDataset(Dataset):
    def __init__(self, image_folder, label_csv, sketch_folder, transform=None):
        self.transform = transform
        self.labels_df = pd.read_csv(label_csv)
        self.labels_df = self.labels_df.dropna()
        existing_images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        filtered_labels = []
        for idx, row in self.labels_df.iterrows():
            image=row['image']+'.jpg'
            if image in existing_images:
                filtered_labels.append(row)
        self.labels_df = pd.DataFrame(filtered_labels)
        
        self.image_folder = image_folder
        self.sketch_folder = sketch_folder
        self.sketch_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Load image
        image_name = self.labels_df.iloc[idx]['image'] + '.jpg'
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Load label
        label = self.labels_df.iloc[idx, 1:].values.astype(float)
        #label_tensor = torch.tensor(label, dtype=torch.float32)
        true_label_image = torch.zeros((256, 256, 7))
        for i, value in enumerate(label):
            true_label_image[:, :, i] = value
        
        
        # Randomly sample sketch
        random_sketch_idx = random.randint(0, len(os.listdir(self.sketch_folder)) - 1)
        sketch_name = os.listdir(self.sketch_folder)[random_sketch_idx]
        sketch_path = os.path.join(self.sketch_folder, sketch_name)
        sketch = Image.open(sketch_path).convert('L')
        transformed_sketch = self.sketch_transform(sketch)
        
        # Generate random label image
        #random_label_image = random_label()
        
        if self.transform:
            image = self.transform(image)
        
        return transformed_sketch, true_label_image, image
    
# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=8, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(8, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        #print(x.shape)
        d1 = self.initial_down(x)
        #print("d1",d1.shape)
        d2 = self.down1(d1)
        #print("d2",d2.shape)
        d3 = self.down2(d2)
        #print("d3",d3.shape)
        d4 = self.down3(d3)
        #print("d4",d4.shape)
        d5 = self.down4(d4)
        #print("d5",d5.shape)
        d6 = self.down5(d5)
        #print("d6",d6.shape)
        d7 = self.down6(d6)
        #print("d7",d7.shape)
        bottleneck = self.bottleneck(d7)
        #print("bottleneck",bottleneck.shape)
        up1 = self.up1(bottleneck)
        #print("up1",up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        #print("up2",up2.shape)
        up3 = self.up3(torch.cat([up2, d6], 1))
        #print("up3",up3.shape)
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        #print(self.final_up(torch.cat([up7, d1], 1)).shape)
        return self.final_up(torch.cat([up7, d1], 1))
    
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=10, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        #print(x.shape,y.shape)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
def save_some_examples(gen, val_loader, epoch, folder):
    x, label, y = next(iter(val_loader))
    x, label, y = x.to(DEVICE), label.to(DEVICE), y.to(DEVICE)
    label=label.permute(0,3,1,2)
    x_c = torch.cat((x, label), dim=1)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x_c)
        print(y_fake.shape)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, label, y) in enumerate(loop):
        x = x.to(DEVICE) #sketches
        label = label.to(DEVICE) #labels
        y = y.to(DEVICE) #images
        
        #print(x.shape, label.shape, y.shape)
        
        label=label.permute(0,3,1,2)
        
        x = torch.cat((x, label), dim=1)
        
        y = torch.cat((y, label), dim=1)
        
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            #print(y_fake.shape)
            y_fake = torch.cat((y_fake, label), dim=1) 
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
            
def main():
    disc = Discriminator(in_channels=9).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        )
        
    train_sketch_folder='/workspace/ConditionalGAN/Dataset/Train/Train_contours'
    train_image_folder='/workspace/ConditionalGAN/Dataset/Train/Train_data'
    train_label_csv='/workspace/ConditionalGAN/Dataset/Train/Train_labels.csv'

    test_sketch_folder='/workspace/ConditionalGAN/Dataset/Test/Test_contours'
    test_image_folder='/workspace/ConditionalGAN/Dataset/Test/Test_data'
    test_label_csv='/workspace/ConditionalGAN/Dataset/Test/Test_Labels.csv'


    train_dataset = ISICDataset(train_image_folder,train_label_csv,train_sketch_folder,transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = ISICDataset(test_image_folder,test_label_csv,test_sketch_folder,transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")
        
if __name__ == "__main__":
    main()

    
test_sketch_folder='/workspace/ConditionalGAN/Dataset/Test/Test_contours'
test_image_folder='/workspace/ConditionalGAN/Dataset/Test/Test_data'
test_label_csv='/workspace/ConditionalGAN/Dataset/Test/Test_Labels.csv'

val_dataset = ISICDataset(test_image_folder,test_label_csv,test_sketch_folder,transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# Load the trained model for inference
G_loaded = Generator(in_channels=3, features=64).to(DEVICE)
D_loaded = Discriminator(in_channels=3).to(DEVICE)

checkpoint2 = torch.load('gen.pth.tar')
G_loaded.load_state_dict(checkpoint2['state_dict'])


import matplotlib.pyplot as plt
from torchvision.utils import make_grid



loop = tqdm(val_loader, leave=True)
# Generate and visualize a single sample image using test loader
for idx, (x, label, y) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE) #sketches
    label = label.to(DEVICE) #labels
    y = y.to(DEVICE) #images

    #print(x.shape, label.shape, y.shape)

    label=label.permute(0,3,1,2)

    x_c = torch.cat((x, label), dim=1)

    # Generate an image using the generator
    generated_image = G_loaded(x_c)

    # Visualize the images side by side
    plt.figure(figsize=(20, 6))

    # Plot original image
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(np.transpose(y[0].cpu().detach().numpy(), (1, 2, 0)))

    # Plot sketch
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Sketch")
    plt.imshow(np.transpose(x[0].cpu().detach().numpy(), (1, 2, 0)), cmap='gray')

    # Plot generated image
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(np.transpose(generated_image[0].cpu().detach().numpy(), (1, 2, 0)))

    plt.show()
    
    if idx==20:
        break


#FID Score
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.linalg import sqrtm

# Define the InceptionV3 model
class InceptionV3(nn.Module):
    def _init_(self):
        super(InceptionV3, self)._init_()
        self.model = models.inception_v3(pretrained=True, aux_logits=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Calculate FID score
def calculate_fid(real_images, generated_images, batch_size=50, device='cuda'):
    # Load InceptionV3 model
    inception_model = InceptionV3().to(device)
    
    # Preprocess images
    def preprocess(images):
        # Resize images to (299, 299) as required by InceptionV3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Scale images to range [-1, 1]
        images = (images - 0.5) * 2
        return images
    
    # Calculate activations for real images
    real_activations = []
    with torch.no_grad():
        for i in range(0, len(real_images), batch_size):
            batch = preprocess(real_images[i:i+batch_size].to(device))
            activations = inception_model(batch)
            real_activations.append(activations.cpu().detach().numpy())
    real_activations = np.concatenate(real_activations, axis=0)

    # Calculate activations for generated images
    generated_activations = []
    with torch.no_grad():
        for i in range(0, len(generated_images), batch_size):
            batch = preprocess(generated_images[i:i+batch_size].to(device))
            activations = inception_model(batch)
            generated_activations.append(activations.cpu().detach().numpy())
    generated_activations = np.concatenate(generated_activations, axis=0)

    # Calculate mean and covariance for real and generated activations
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    # Numerical adjustment for covariance matrices
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_gen += eps * np.eye(sigma_gen.shape[0])

    # Calculate Frechet distance
    sqrt_sigma_real = sqrtm(sigma_real)
    sqrt_sigma_real_sigma_gen_sqrt = sqrtm(sqrt_sigma_real @ sigma_gen @ sqrt_sigma_real)
    if np.iscomplexobj(sqrt_sigma_real_sigma_gen_sqrt):
        sqrt_sigma_real_sigma_gen_sqrt = sqrt_sigma_real_sigma_gen_sqrt.real
    fid = np.linalg.norm(mu_real - mu_gen) + np.trace(sigma_real + sigma_gen - 2 * sqrt_sigma_real_sigma_gen_sqrt)

    return fid


# Generate and visualize a single sample image using test loader
for idx, (x, label, y) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE) #sketches
    label = label.to(DEVICE) #labels
    y = y.to(DEVICE) #images

    #print(x.shape, label.shape, y.shape)

    label=label.permute(0,3,1,2)

    x_c = torch.cat((x, label), dim=1)

    # Generate an image using the generator
    generated_image = G_loaded(x_c)

    # Calculate FID score
    fid_score = calculate_fid(y, generated_image,device='gpu')
    print("FID Score:", fid_score)

    #Inception Score
    import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import numpy as np

loop = tqdm(val_loader, leave=True)

# Function to calculate Inception Score
def inception_score(images, batch_size=32, splits=10):
    
    # Load pre-trained InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False).eval()

    # Define transformation for images
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Calculate conditional label distribution p(y|x) for generated images
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = torch.stack([preprocess(img) for img in batch], dim=0)
            pred = torch.nn.functional.softmax(model(batch), dim=1).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    # Calculate marginal probability distribution p(y)
    p_y = np.mean(preds, axis=0)

    # Calculate KL divergence between p(y|x) and p(y)
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        q_y_x = np.mean(part, axis=0)
        kl_d = entropy(p_y, q_y_x)
        scores.append(kl_d)
    
    # Calculate Inception Score
    scores = np.array(scores)
    is_score = np.exp(np.mean(scores))

    return is_score

# Generate and visualize a single sample image using test loader
for idx, (x, label, y) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE) #sketches
    label = label.to(DEVICE) #labels
    y = y.to(DEVICE) #images

    #print(x.shape, label.shape, y.shape)

    label=label.permute(0,3,1,2)

    x_c = torch.cat((x, label), dim=1)

    # Calculate Inception Score
    is_score = inception_score(x_c)

    print("Inception Score:", is_score)


""" Architecture 2 """

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import pandas as pd
import random
import wandb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
L1_LAMBDA = 100
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "discpaierd.pth.tar"
CHECKPOINT_GEN = "genpaired.pth.tar"


wandb.init(project="ConditionalGAN")

class ISICDataset(Dataset):
    def __init__(self, img_dir, sketch_dir, label_csv, transform=None):
        self.img_dir = img_dir
        self.sketch_dir = sketch_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_csv)
        
        # Get list of filenames (assuming equal number of sketches and images)
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        base_filename = os.path.splitext(self.filenames[index])[0]
        
        # Construct paths
        sketch_path = os.path.join(self.sketch_dir, base_filename + "_segmentation.png")
        real_image_path = os.path.join(self.img_dir, base_filename + ".jpg")
        
         # Load label
        label = self.labels_df.iloc[index, 1:].values.astype(float)
        #label_tensor = torch.tensor(label, dtype=torch.float32)
        true_label_image = torch.zeros((256, 256, 7))
        for i, value in enumerate(label):
            true_label_image[:, :, i] = value

        # Load images
        sketch = Image.open(sketch_path)
        real_image = Image.open(real_image_path)

        # Resize
        sketch = sketch.convert('RGB')

        input_image = np.array(sketch)
        target_image = np.array(real_image)

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image, true_label_image

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, label_dim=7,features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels+label_dim, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        #print(x.shape)
        d1 = self.initial_down(x)
        #print("d1",d1.shape)
        d2 = self.down1(d1)
        #print("d2",d2.shape)
        d3 = self.down2(d2)
        #print("d3",d3.shape)
        d4 = self.down3(d3)
        #print("d4",d4.shape)
        d5 = self.down4(d4)
        #print("d5",d5.shape)
        d6 = self.down5(d5)
        #print("d6",d6.shape)
        d7 = self.down6(d6)
        #print("d7",d7.shape)
        bottleneck = self.bottleneck(d7)
        #print("bottleneck",bottleneck.shape)
        up1 = self.up1(bottleneck)
        #print("up1",up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        #print("up2",up2.shape)
        up3 = self.up3(torch.cat([up2, d6], 1))
        #print("up3",up3.shape)
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        #print(self.final_up(torch.cat([up7, d1], 1)).shape)
        return self.final_up(torch.cat([up7, d1], 1))
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, label_dim=7, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                (in_channels+label_dim) * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        #print(x.shape,y.shape)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
def save_some_examples(gen, val_loader, epoch, folder):
    x, y, label = next(iter(val_loader))
    x, label, y = x.to(DEVICE), label.to(DEVICE), y.to(DEVICE)
    label=label.permute(0,3,1,2)
    
    x_c = torch.cat((x, label), dim=1)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x_c)
        #print(y_fake.shape)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y, label) in enumerate(loop):
        x = x.to(DEVICE) #sketches
        label = label.to(DEVICE) #labels
        y = y.to(DEVICE) #images
        
        #print(x.shape, label.shape, y.shape)
        
        label=label.permute(0,3,1,2)
        
        #print(x.shape, label.shape, y.shape)
        
        x_c = torch.cat((x, label), dim=1)
        
        y_c = torch.cat((y, label), dim=1)
        
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x_c)
            #print(y_fake.shape)
            y_fake = torch.cat((y_fake, label), dim=1) 
            D_real = disc(x_c, y_c)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            #print(x.shape, y_fake.shape)
            D_fake = disc(x_c, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x_c, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y_c) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        wandb.log({"epoch": idx+1, "D_loss": D_loss.item(), "G_loss": G_loss.item()})

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
            
            
 def main():
    disc = Discriminator(in_channels=3, label_dim=7).to(DEVICE)
    gen = Generator(in_channels=3, label_dim=7, features=64).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        )
        
    train_sketch_folder='/workspace/ConditionalGAN/New_Dataset/Train/Paired_train_sketches'
    train_image_folder='/workspace/ConditionalGAN/New_Dataset/Train/Train_data'
    train_label_csv='/workspace/ConditionalGAN/New_Dataset/Train/Train_labels.csv'

    test_sketch_folder='/workspace/ConditionalGAN/New_Dataset/Test/Paired_test_sketch'
    test_image_folder='/workspace/ConditionalGAN/New_Dataset/Test/Test_data'
    test_label_csv='/workspace/ConditionalGAN/New_Dataset/Test/Test_labels.csv'


    train_dataset = ISICDataset(train_image_folder,train_sketch_folder,train_label_csv)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = ISICDataset(test_image_folder,test_sketch_folder,test_label_csv)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation-paired")
    wandb.finish()

if __name__ == "__main__":
    main()
    
test_sketch_folder='/workspace/ConditionalGAN/New_Dataset/Test/Paired_test_sketch'
test_image_folder='/workspace/ConditionalGAN/New_Dataset/Test/Test_data'
test_label_csv='/workspace/ConditionalGAN/New_Dataset/Test/Test_labels.csv'

val_dataset = ISICDataset(test_image_folder,test_sketch_folder,test_label_csv)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load the trained model for inference
G_loaded = Generator(in_channels=3, label_dim=7, features=64).to(DEVICE)
D_loaded = Discriminator(in_channels=3).to(DEVICE)

checkpoint1 = torch.load('disc-paired.pth.tar')
checkpoint2 = torch.load('gen-paired.pth.tar')
D_loaded.load_state_dict(checkpoint1['state_dict'])
G_loaded.load_state_dict(checkpoint2['state_dict'])


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
os.makedirs("images", exist_ok=True)
loop = tqdm(val_loader, leave=True)
# Generate and visualize a single sample image using test loader
for idx, (x, y, label) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE)  # sketch
    label = label.to(DEVICE)  # label
    y = y.to(DEVICE)  # original image

    label = label.permute(0, 3, 1, 2)

    # Concatenate input sketch and label
    x_c = torch.cat((x, label), dim=1)

    # Generate an image using the generator
    generated_image = G_loaded(x_c)
    
    generated_image_np = np.transpose(generated_image[0].cpu().detach().numpy(), (1, 2, 0))
    plt.imsave(f"images/generated_image_{idx}.png", generated_image_np)

    # Visualize the images side by side
    plt.figure(figsize=(20, 6))

    # Plot original image
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(np.transpose(y[0].cpu().detach().numpy(), (1, 2, 0)))

    # Plot sketch
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Sketch")
    plt.imshow(np.transpose(x[0].cpu().detach().numpy(), (1, 2, 0)), cmap='gray')

    # Plot generated image
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(np.transpose(generated_image[0].cpu().detach().numpy(), (1, 2, 0)))

    plt.show()
    
    if idx==20:
        break

## Saving the images for classification testing
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

os.makedirs("images", exist_ok=True)

# Loop over the validation data loader
for idx, (x, y, label) in enumerate(val_loader):
    # Move data to device
    x = x.to(DEVICE)  # sketch
    label = label.to(DEVICE)  # label
    y = y.to(DEVICE)  # original image

    label = label.permute(0, 3, 1, 2)

    # Concatenate input sketch and label
    x_c = torch.cat((x, label), dim=1)

    # Generate an image using the generator
    generated_image = G_loaded(x_c)

    # Convert the generated image to numpy array
    generated_image_np = generated_image[0].cpu().detach().numpy()
    
    # Normalize the pixel values in the range [0, 1]
    generated_image_np_normalized = (generated_image_np - generated_image_np.min()) / (generated_image_np.max() - generated_image_np.min())

    # Get the corresponding filename
    filename = val_loader.dataset.filenames[idx]

    # Save the normalized image with the original filename
    plt.imsave(f"images/{filename}", np.transpose(generated_image_np_normalized, (1, 2, 0)))


#FID Score
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.linalg import sqrtm

# Define the InceptionV3 model
class InceptionV3(nn.Module):
    def _init_(self):
        super(InceptionV3, self)._init_()
        self.model = models.inception_v3(pretrained=True, aux_logits=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Calculate FID score
def calculate_fid(real_images, generated_images, batch_size=50, device='cuda'):
    # Load InceptionV3 model
    inception_model = InceptionV3().to(device)
    
    # Preprocess images
    def preprocess(images):
        # Resize images to (299, 299) as required by InceptionV3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Scale images to range [-1, 1]
        images = (images - 0.5) * 2
        return images
    
    # Calculate activations for real images
    real_activations = []
    with torch.no_grad():
        for i in range(0, len(real_images), batch_size):
            batch = preprocess(real_images[i:i+batch_size].to(device))
            activations = inception_model(batch)
            real_activations.append(activations.cpu().detach().numpy())
    real_activations = np.concatenate(real_activations, axis=0)

    # Calculate activations for generated images
    generated_activations = []
    with torch.no_grad():
        for i in range(0, len(generated_images), batch_size):
            batch = preprocess(generated_images[i:i+batch_size].to(device))
            activations = inception_model(batch)
            generated_activations.append(activations.cpu().detach().numpy())
    generated_activations = np.concatenate(generated_activations, axis=0)

    # Calculate mean and covariance for real and generated activations
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    # Numerical adjustment for covariance matrices
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_gen += eps * np.eye(sigma_gen.shape[0])

    # Calculate Frechet distance
    sqrt_sigma_real = sqrtm(sigma_real)
    sqrt_sigma_real_sigma_gen_sqrt = sqrtm(sqrt_sigma_real @ sigma_gen @ sqrt_sigma_real)
    if np.iscomplexobj(sqrt_sigma_real_sigma_gen_sqrt):
        sqrt_sigma_real_sigma_gen_sqrt = sqrt_sigma_real_sigma_gen_sqrt.real
    fid = np.linalg.norm(mu_real - mu_gen) + np.trace(sigma_real + sigma_gen - 2 * sqrt_sigma_real_sigma_gen_sqrt)

    return fid


# Generate and visualize a single sample image using test loader
for idx, (x, y, label) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE)  # sketch
    label = label.to(DEVICE)  # label
    y = y.to(DEVICE)  # original image

    label = label.permute(0, 3, 1, 2)

    # Concatenate input sketch and label
    x_c = torch.cat((x, label), dim=1)

    # Generate an image using the generator
    generated_image = G_loaded(x_c)

    # Calculate FID score
    fid_score = calculate_fid(y, generated_image,device='gpu')
    print("FID Score:", fid_score)

#Inception Score
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import numpy as np

loop = tqdm(val_loader, leave=True)

# Function to calculate Inception Score
def inception_score(images, batch_size=32, splits=10):
    
    # Load pre-trained InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False).eval()

    # Define transformation for images
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Calculate conditional label distribution p(y|x) for generated images
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = torch.stack([preprocess(img) for img in batch], dim=0)
            pred = torch.nn.functional.softmax(model(batch), dim=1).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    # Calculate marginal probability distribution p(y)
    p_y = np.mean(preds, axis=0)

    # Calculate KL divergence between p(y|x) and p(y)
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        q_y_x = np.mean(part, axis=0)
        kl_d = entropy(p_y, q_y_x)
        scores.append(kl_d)
    
    # Calculate Inception Score
    scores = np.array(scores)
    is_score = np.exp(np.mean(scores))

    return is_score

# Generate and visualize a single sample image using test loader
for idx, (x, y, label) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE)  # sketch
    label = label.to(DEVICE)  # label
    y = y.to(DEVICE)  # original image

    label = label.permute(0, 3, 1, 2)

    # Concatenate input sketch and label
    x_c = torch.cat((x, label), dim=1)

    # Calculate Inception Score
    is_score = inception_score(x_c)

    print("Inception Score:", is_score)

"""
Architecture 3"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

class ISICDataset(Dataset):
    def __init__(self, img_dir, sketch_dir, transform=None):
        self.img_dir = img_dir
        self.sketch_dir = sketch_dir
        self.transform = transform
        
        # Get list of filenames (assuming equal number of sketches and images)
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        base_filename = os.path.splitext(self.filenames[index])[0]
        
        # Construct paths
        sketch_path = os.path.join(self.sketch_dir, base_filename + "_segmentation.png")
        real_image_path = os.path.join(self.img_dir, base_filename + ".jpg")

        # Load images
        sketch = Image.open(sketch_path)
        real_image = Image.open(real_image_path)

        # Resize
        sketch = sketch.convert('RGB')

        input_image = np.array(sketch)
        target_image = np.array(real_image)

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        print(x.shape)
        d1 = self.initial_down(x)
        print("d1",d1.shape)
        d2 = self.down1(d1)
        print("d2",d2.shape)
        d3 = self.down2(d2)
        print("d3",d3.shape)
        d4 = self.down3(d3)
        print("d4",d4.shape)
        d5 = self.down4(d4)
        print("d5",d5.shape)
        d6 = self.down5(d5)
        print("d6",d6.shape)
        d7 = self.down6(d6)
        print("d7",d7.shape)
        bottleneck = self.bottleneck(d7)
        print("bottleneck",bottleneck.shape)
        up1 = self.up1(bottleneck)
        print("up1",up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        print("up2",up2.shape)
        up3 = self.up3(torch.cat([up2, d6], 1))
        print("up3",up3.shape)
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))
    
    
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
    
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        )

    train_dataset = ISICDataset(img_dir=r'/workspace/ConditionalGAN/New_Dataset/Train/Train_data', sketch_dir=r'/workspace/ConditionalGAN/New_Dataset/Train/Paired_train_sketches')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = ISICDataset(img_dir=r'/workspace/ConditionalGAN/New_Dataset/Test/Test_data', sketch_dir=r'/workspace/ConditionalGAN/New_Dataset/Test/Paired_test_sketch')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")
        
if __name__ == "__main__":
    main()
    
    
    
test_sketch_folder='/workspace/ConditionalGAN/New_Dataset/Test/Paired_test_sketch'
test_image_folder='/workspace/ConditionalGAN/New_Dataset/Test/Test_data'
test_label_csv='/workspace/ConditionalGAN/New_Dataset/Test/Test_labels.csv'

val_dataset = ISICDataset(test_image_folder,test_sketch_folder,test_label_csv)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



# Load the trained model for inference
G_loaded = Generator(in_channels=3, features=64).to(DEVICE)
D_loaded = Discriminator(in_channels=3).to(DEVICE)

checkpoint1 = torch.load('disc.pth1.tar')
checkpoint2 = torch.load('gen.pth1.tar')
D_loaded.load_state_dict(checkpoint1['state_dict'])
G_loaded.load_state_dict(checkpoint2['state_dict'])


import matplotlib.pyplot as plt
from torchvision.utils import make_grid

loop = tqdm(val_loader, leave=True)
# Generate and visualize a single sample image using test loader
for idx, (x, y) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE)  # sketch
    y = y.to(DEVICE)  # original image

    # Generate an image using the generator
    generated_image = G_loaded(x)

    # Visualize the images side by side
    plt.figure(figsize=(20, 6))

    # Plot original image
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(np.transpose(y[0].cpu().detach().numpy(), (1, 2, 0)))

    # Plot sketch
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Sketch")
    plt.imshow(np.transpose(x[0].cpu().detach().numpy(), (1, 2, 0)), cmap='gray')

    # Plot generated image
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Generated Image")
    plt.imshow(np.transpose(generated_image[0].cpu().detach().numpy(), (1, 2, 0)))

    plt.show()
    
    if idx==20:
        break
#FID Score
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.linalg import sqrtm

# Define the InceptionV3 model
class InceptionV3(nn.Module):
    def _init_(self):
        super(InceptionV3, self)._init_()
        self.model = models.inception_v3(pretrained=True, aux_logits=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Calculate FID score
def calculate_fid(real_images, generated_images, batch_size=50, device='cuda'):
    # Load InceptionV3 model
    inception_model = InceptionV3().to(device)
    
    # Preprocess images
    def preprocess(images):
        # Resize images to (299, 299) as required by InceptionV3
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # Scale images to range [-1, 1]
        images = (images - 0.5) * 2
        return images
    
    # Calculate activations for real images
    real_activations = []
    with torch.no_grad():
        for i in range(0, len(real_images), batch_size):
            batch = preprocess(real_images[i:i+batch_size].to(device))
            activations = inception_model(batch)
            real_activations.append(activations.cpu().detach().numpy())
    real_activations = np.concatenate(real_activations, axis=0)

    # Calculate activations for generated images
    generated_activations = []
    with torch.no_grad():
        for i in range(0, len(generated_images), batch_size):
            batch = preprocess(generated_images[i:i+batch_size].to(device))
            activations = inception_model(batch)
            generated_activations.append(activations.cpu().detach().numpy())
    generated_activations = np.concatenate(generated_activations, axis=0)

    # Calculate mean and covariance for real and generated activations
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    # Numerical adjustment for covariance matrices
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_gen += eps * np.eye(sigma_gen.shape[0])

    # Calculate Frechet distance
    sqrt_sigma_real = sqrtm(sigma_real)
    sqrt_sigma_real_sigma_gen_sqrt = sqrtm(sqrt_sigma_real @ sigma_gen @ sqrt_sigma_real)
    if np.iscomplexobj(sqrt_sigma_real_sigma_gen_sqrt):
        sqrt_sigma_real_sigma_gen_sqrt = sqrt_sigma_real_sigma_gen_sqrt.real
    fid = np.linalg.norm(mu_real - mu_gen) + np.trace(sigma_real + sigma_gen - 2 * sqrt_sigma_real_sigma_gen_sqrt)

    return fid


for idx, (x, y) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE)  # sketch
    y = y.to(DEVICE)  # original image

    # Generate an image using the generator
    generated_image = G_loaded(x)

    # Calculate FID score
    fid_score = calculate_fid(y, generated_image,device='gpu')
    print("FID Score:", fid_score)


#Inception Score
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import numpy as np

loop = tqdm(val_loader, leave=True)

# Function to calculate Inception Score
def inception_score(images, batch_size=32, splits=10):
    
    # Load pre-trained InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False).eval()

    # Define transformation for images
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # Calculate conditional label distribution p(y|x) for generated images
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = torch.stack([preprocess(img) for img in batch], dim=0)
            pred = torch.nn.functional.softmax(model(batch), dim=1).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    # Calculate marginal probability distribution p(y)
    p_y = np.mean(preds, axis=0)

    # Calculate KL divergence between p(y|x) and p(y)
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        q_y_x = np.mean(part, axis=0)
        kl_d = entropy(p_y, q_y_x)
        scores.append(kl_d)
    
    # Calculate Inception Score
    scores = np.array(scores)
    is_score = np.exp(np.mean(scores))

    return is_score

for idx, (x, y) in enumerate(loop):
    #test_iter = iter(val_loader)
    #x, y, label = next(test_iter)

    x = x.to(DEVICE)  # sketch
    y = y.to(DEVICE)  # original image

    # Generate an image using the generator
    generated_image = G_loaded(x)

    # Calculate Inception Score
    is_score = inception_score(generated_image)

    print("Inception Score:", is_score)

        
""" Classifier Model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


class SkinDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name+'.jpg')
        labels = self.labels_frame.iloc[idx, 1:].values
        #labels = torch.argmax(torch.tensor(labels.astype('float')))
        #labels = torch.argmax(torch.tensor(labels.astype('float'))).unsqueeze(0)
        labels = torch.argmax(torch.tensor(labels.astype('float'))).unsqueeze(0).squeeze()


        
        if self.transform:
            image = self.transform(image)

        return image, labels



class SkinDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseClassifier, self).__init__()
        #base =models.resnet152(pretrained=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        #x = self.pool(self.relu(self.bn4(self.conv4(x))))
        #print("X.shape")
        #print(x.shape)
        #x = x.view(-1, 256 * 28 * 28)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_labels_df = pd.read_csv('/workspace/ConditionalGAN/Dataset/Train/Train_labels.csv')
test_labels_df = pd.read_csv('/workspace/ConditionalGAN/Dataset/Test/Test_Labels.csv')


label_encoder = LabelEncoder()
train_labels_df['label'] = label_encoder.fit_transform(train_labels_df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values.argmax(axis=1))
test_labels_df['label'] = label_encoder.transform(test_labels_df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values.argmax(axis=1))

# Start
train_dataset = SkinDiseaseDataset(csv_file='/workspace/ConditionalGAN/Dataset/Train/Train_labels.csv', root_dir='/workspace/ConditionalGAN/Dataset/Train/Train_data', transform=transform)
test_dataset = SkinDiseaseDataset(csv_file='/workspace/ConditionalGAN/Dataset/Test/Test_Labels.csv', root_dir='/workspace/ConditionalGAN/Dataset/Test/Test_data', transform=transform)

#end

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True ,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False , drop_last=True)


lr = 0.001
num_epochs = 15
num_classes = 7

classifier = SkinDiseaseClassifier(num_classes)


optimizer = optim.Adam(classifier.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier.to(device)
criterion.to(device)


for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()


        outputs = classifier(images)
        
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track the accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluate the classifier
classifier.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = classifier(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the trained classifier
torch.save(classifier.state_dict(), 'skin_disease_classifier.pth')


import torchvision.models as models

# Load the saved state dictionary
checkpoint = torch.load('skin_disease_classifier.pth')

# Load the state dictionary into the model
classifier.load_state_dict(checkpoint)


test_dataset = SkinDiseaseDataset(csv_file='/workspace/ConditionalGAN/Dataset/Test/Test_Labels.csv', root_dir='/workspace/ConditionalGAN/images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False , drop_last=True)


# Evaluate the classifier
classifier.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = classifier(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')

 