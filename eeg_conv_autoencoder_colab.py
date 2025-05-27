import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CHANNELS = 23
INPUT_LENGTH = 10240
LATENT_DIM = 128
BATCH_SIZE = 32 
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX_LR_SCHEDULER = 50
ETA_MIN_LR_SCHEDULER = 1e-5
VAL_SPLIT = 0.2 # Validation split ratio
DUMMY_NPY_FILENAME = "dummy_eeg_data.npy"
ENCODER_SAVE_PATH = "encoder.pt"
LATENT_FEATURES_SAVE_PATH = "latent.npy"

# --- Dataset ---
class EEGDataset(Dataset):
    """
    Dataset class for loading EEG data from a .npy file and applying per-channel MinMaxScaler.
    """
    def __init__(self, npy_file_path):
        raw_data = np.load(npy_file_path).astype(np.float32) # Expected shape: (n_samples, 23, 10240)
        
        if raw_data.ndim != 3 or raw_data.shape[1] != N_CHANNELS or raw_data.shape[2] != INPUT_LENGTH:
            raise ValueError(f"Input data has shape {raw_data.shape}, expected ({raw_data.shape[0]}, {N_CHANNELS}, {INPUT_LENGTH})")

        # Apply per-channel MinMaxScaler to [0,1]
        # For each channel, find its min/max across all samples and time points, then scale.
        scaled_data = np.zeros_like(raw_data)
        for i in range(raw_data.shape[1]):  # Iterate over channels
            channel_content = raw_data[:, i, :] # Shape (n_samples, 10240)
            min_val = np.min(channel_content)
            max_val = np.max(channel_content)
            
            if max_val == min_val:
                # If all values in a channel are the same, map them to 0 
                # (assuming feature_range starts at 0 for MinMaxScaler)
                scaled_data[:, i, :] = np.zeros_like(channel_content, dtype=np.float32)
            else:
                scaled_data[:, i, :] = (channel_content - min_val) / (max_val - min_val)
        self.data = scaled_data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

# --- Encoder ---
class Encoder(nn.Module):
    """
    Encoder part of the Conv-AutoEncoder.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # 10240 -> 5120
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # 5120 -> 2560
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # 2560 -> 1280
        )
        # Output of conv_block3: (B, 128, 1280)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # (B, 128, 1280) -> (B, 128, 1)
        self.flatten = nn.Flatten() # (B, 128, 1) -> (B, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, LATENT_DIM) # (B, 128) -> (B, LATENT_DIM)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --- Decoder ---
class Decoder(nn.Module):
    """
    Decoder part of the Conv-AutoEncoder.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # Input to fc: (B, LATENT_DIM)
        # Output should be shaped to (B, 128, 1280) to match encoder's pre-pooling state
        self.fc = nn.Linear(LATENT_DIM, 128 * 1280) 
        
        # The following ConvTranspose1d layers are designed to take an input of (B, 128, 1280)
        # and progressively upsample it to (B, N_CHANNELS, INPUT_LENGTH)

        # Input to deconv_block1 will be (B, 128, 1280)
        self.deconv_block1 = nn.Sequential(
            # L_out = (L_in - 1)*stride - 2*padding + kernel_size + output_padding
            # (1280-1)*2 - 2*2 + 5 + 1 = 2558 - 4 + 5 + 1 = 2560
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # (B,128,1280) -> (B,64,2560)
            nn.ReLU()
        )
        self.deconv_block2 = nn.Sequential(
            # (2560-1)*2 - 2*3 + 7 + 1 = 5118 - 6 + 7 + 1 = 5120
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1), # (B,64,2560) -> (B,32,5120)
            nn.ReLU()
        )
        self.deconv_block3 = nn.Sequential(
            # (5120-1)*2 - 2*3 + 7 + 1 = 10238 - 6 + 7 + 1 = 10240
            nn.ConvTranspose1d(32, N_CHANNELS, kernel_size=7, stride=2, padding=3, output_padding=1), # (B,32,5120) -> (B,23,10240)
            nn.Sigmoid() # To match 0-1 scale of input
        )

    def forward(self, x): # x is (B, LATENT_DIM)
        x = self.fc(x)   # (B, 128 * 1280)
        # Reshape to (B, 128 channels, 1280 length)
        x = x.view(x.size(0), 128, 1280) 
        
        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = self.deconv_block3(x) # Output: (B, N_CHANNELS, 10240)
        return x

# --- AutoEncoder ---
class ConvAutoEncoder(nn.Module):
    """
    Full Convolutional AutoEncoder model.
    """
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# --- Training Function ---
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, encoder_save_path):
    best_val_loss = float('inf')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        for batch_idx, data_batch in enumerate(train_loader):
            inputs = data_batch.to(device) # Shape: (B, N_CHANNELS, INPUT_LENGTH)
            
            optimizer.zero_grad()
            reconstructed = model(inputs) # Shape: (B, N_CHANNELS, INPUT_LENGTH)

            # Target for loss is now the same as inputs
            target_for_loss = inputs
            
            loss = criterion(reconstructed, target_for_loss)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * inputs.size(0)
        
        train_loss_epoch /= len(train_loader.dataset)

        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for data_batch in val_loader:
                inputs = data_batch.to(device)
                reconstructed = model(inputs)
                
                # Target for loss is now the same as inputs
                target_for_loss = inputs
                
                loss = criterion(reconstructed, target_for_loss)
                val_loss_epoch += loss.item() * inputs.size(0)
        
        val_loss_epoch /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss_epoch:.6f} | Val MSE: {val_loss_epoch:.6f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.encoder.state_dict(), encoder_save_path)
            print(f"Best encoder saved to {encoder_save_path} (Val MSE: {best_val_loss:.6f})")
        
        if scheduler:
            scheduler.step()
    
    return best_val_loss

# --- Latent Extraction Function ---
def extract_latent_features(encoder_model, data_loader, device, output_path):
    encoder_model.to(device)
    encoder_model.eval()
    
    all_latents_list = []
    with torch.no_grad():
        for data_batch in data_loader:
            inputs = data_batch.to(device)
            latents = encoder_model(inputs) # Shape: (B, LATENT_DIM)
            all_latents_list.append(latents.cpu().numpy())
            
    all_latents_np = np.concatenate(all_latents_list, axis=0)
    np.save(output_path, all_latents_np)
    print(f"Latent features saved to {output_path}, shape: {all_latents_np.shape}")
    return all_latents_np

# --- Main Execution ---
def main():
    print(f"Using device: {DEVICE}")

    # 1. Create a dummy .npy file for demonstration
    # This part should be replaced with user's actual data loading path
    DUMMY_N_SAMPLES = 100
    dummy_data_array = np.random.rand(DUMMY_N_SAMPLES, N_CHANNELS, INPUT_LENGTH).astype(np.float32)
    np.save(DUMMY_NPY_FILENAME, dummy_data_array)
    print(f"Generated dummy data and saved to {DUMMY_NPY_FILENAME} with shape {dummy_data_array.shape}")

    # 2. Dataset and DataLoader
    try:
        full_dataset = EEGDataset(npy_file_path=DUMMY_NPY_FILENAME)
    except ValueError as e:
        print(f"Error creating dataset: {e}")
        if os.path.exists(DUMMY_NPY_FILENAME):
            os.remove(DUMMY_NPY_FILENAME)
        return

    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    if train_size <= 0 or val_size <=0:
        print(f"Dataset too small for splitting with {VAL_SPLIT*100}% validation. Needs at least {1/VAL_SPLIT} samples.")
        if os.path.exists(DUMMY_NPY_FILENAME): os.remove(DUMMY_NPY_FILENAME)
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    num_workers = 2 if DEVICE.type == 'cuda' else 0 # Basic num_workers logic
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    # DataLoader for full dataset for latent extraction (no shuffle)
    full_data_loader_for_extraction = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    print(f"DataLoaders created: Train {len(train_dataset)} samples, Val {len(val_dataset)} samples.")

    # 3. Model, Optimizer, Criterion, Scheduler
    autoencoder_model = ConvAutoEncoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX_LR_SCHEDULER, eta_min=ETA_MIN_LR_SCHEDULER)

    # Optional: Print model summary (requires torchsummary pip install torchsummary)
    # try:
    #     from torchsummary import summary
    #     print("\nEncoder Summary:")
    #     summary(autoencoder_model.encoder, (N_CHANNELS, INPUT_LENGTH), device=DEVICE.type)
    #     print("\nDecoder Summary (Note: input for summary is (LATENT_DIM,) assuming direct latent vector input):")
    #     # For decoder summary, the direct input is the latent vector from encoder
    #     # summary(autoencoder_model.decoder, (LATENT_DIM,), device=DEVICE.type) # This might need adjustment based on how decoder handles view/unflatten
    #     print("\nAutoEncoder Summary:")
    #     summary(autoencoder_model, (N_CHANNELS, INPUT_LENGTH), device=DEVICE.type)
    # except ImportError:
    #     print("torchsummary not found. Skipping model summary.")
    #     print("You can install it via 'pip install torchsummary'")
    # except Exception as e:
    #     print(f"Error during model summary: {e}")


    # 4. Train the model
    print("Starting training...")
    train_model(autoencoder_model, train_loader, val_loader, optimizer, scheduler, criterion, EPOCHS, DEVICE, ENCODER_SAVE_PATH)
    print("Training finished.")

    # 5. Extract latent features using the best saved encoder
    print("Extracting latent features...")
    # Create a new encoder instance and load the saved best weights
    best_encoder_instance = Encoder().to(DEVICE)
    try:
        best_encoder_instance.load_state_dict(torch.load(ENCODER_SAVE_PATH, map_location=DEVICE))
        print(f"Successfully loaded best encoder weights from {ENCODER_SAVE_PATH}")
        extract_latent_features(best_encoder_instance, full_data_loader_for_extraction, DEVICE, LATENT_FEATURES_SAVE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Encoder weights file '{ENCODER_SAVE_PATH}' not found. Skipping latent feature extraction.")
    except Exception as e:
        print(f"ERROR loading encoder weights or extracting features: {e}")
        
    print("Latent feature extraction process completed.")

    # Clean up dummy file
    if os.path.exists(DUMMY_NPY_FILENAME):
        os.remove(DUMMY_NPY_FILENAME)
        print(f"Dummy data file {DUMMY_NPY_FILENAME} removed.")

    print("Script execution completed.")

if __name__ == '__main__':
    main() 