import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import random
import copy
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Constants and Configuration
CONFIG = {
    'data_root': '/kaggle/input/celeb-df-v2',
    'batch_size': 4,
    'num_frames': 20,
    'frame_size': 224, # EfficientNet-B0 default input size
    'num_epochs': 10,
    'learning_rate': 1e-4, # May need adjustment for EfficientNet (try 5e-5 or 1e-5 if unstable)
    'weight_decay': 1e-5, # May need adjustment
    'lstm_hidden_dim': 512,
    'dropout_rate': 0.5,
    'use_attention': True,
    'use_spatial_dropout': True, # Renamed, applied after feature extraction before LSTM
    'use_temporal_dropout': True, # Dropout within LSTM layers
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'early_stopping_patience': 5,
    'model_save_path': './model_checkpoints_efficientnet', # Separate directory
    'num_workers': 2, # Reduced further just in case of loader issues
    'layers_to_freeze': 3, # Number of initial feature blocks in EfficientNet to freeze (0=stem, 1=stage1, etc.)
}

# Make sure model checkpoint directory exists
os.makedirs(CONFIG['model_save_path'], exist_ok=True)

# Helper Functions (Identical to previous versions)
def extract_frames(video_path, num_frames):
    """Extract evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        # Return blank frames if video cannot be opened
        return np.zeros((num_frames, CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video {video_path} has zero frames.")
        cap.release()
        return np.zeros((num_frames, CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)

    if total_frames <= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, endpoint=True, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, endpoint=True, dtype=int)

    frames = []
    processed_indices = set() # To handle cases where linspace gives duplicate indices for low total_frames

    for idx in indices:
        if idx in processed_indices: # If index already processed (due to duplication), try next available
             original_idx = idx
             while idx in processed_indices and idx < total_frames - 1:
                 idx += 1
             # If we exhausted options, just reuse the original (might happen if num_frames > total_frames)
             if idx in processed_indices:
                 idx = original_idx # Fallback to original index if no other unique frame is available

        if idx >= total_frames: # Boundary check
              idx = total_frames - 1

        if idx not in processed_indices:
             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
             ret, frame = cap.read()
             if ret:
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 frame = cv2.resize(frame, (CONFIG['frame_size'], CONFIG['frame_size']))
                 frames.append(frame)
                 processed_indices.add(idx)
             else:
                 # Attempt to read the previous frame if reading fails
                 if idx > 0:
                     cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
                     ret, frame = cap.read()
                     if ret:
                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                         frame = cv2.resize(frame, (CONFIG['frame_size'], CONFIG['frame_size']))
                         frames.append(frame)
                         processed_indices.add(idx) # Still mark idx as processed conceptually
                     else:
                         frames.append(np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8))
                         processed_indices.add(idx)
                 else: # If first frame fails, add zeros
                    frames.append(np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8))
                    processed_indices.add(idx)


    cap.release()

    # Pad if fewer frames were extracted than needed
    while len(frames) < num_frames:
        if frames: # Pad with the last valid frame if available
            frames.append(frames[-1].copy())
        else: # Pad with zeros if no frames were read at all
            frames.append(np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8))

    # Ensure all frames have the correct shape just in case
    final_frames = []
    for i in range(len(frames)):
        frame = frames[i]
        if frame.shape != (CONFIG['frame_size'], CONFIG['frame_size'], 3):
             try:
                 frame = cv2.resize(frame, (CONFIG['frame_size'], CONFIG['frame_size']))
             except Exception as resize_e:
                 print(f"Error resizing frame {i} from video {video_path}. Shape was {frame.shape}. Error: {resize_e}")
                 frame = np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8) # Use blank frame on error
        final_frames.append(frame)


    return np.array(final_frames[:num_frames]) # Ensure exactly num_frames are returned


def get_class_weights(labels):
    """Calculate class weights inversely proportional to class frequencies."""
    class_counts = np.bincount(labels)
    # Handle potential zero counts if a class is missing in the training split (unlikely with stratify)
    if len(class_counts) < 2: # Assuming binary classification
        return torch.FloatTensor([1.0, 1.0])
    if class_counts[0] == 0 or class_counts[1] == 0:
        print("Warning: One class has zero samples in the training set.")
        # Assign equal weight or handle as appropriate
        return torch.FloatTensor([1.0, 1.0])

    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)

def plot_confusion_matrix(cm, epoch=None, save_path=CONFIG['model_save_path']):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted Fake', 'Predicted Real'], yticklabels=['True Fake', 'True Real'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    title = f'Confusion Matrix'
    if epoch is not None:
        title += f' - Epoch {epoch+1}'
    plt.title(title)
    if epoch is not None:
        plt.savefig(os.path.join(save_path, f'confusion_matrix_epoch_{epoch+1}.png'))
    plt.close()

def plot_metrics(metrics, save_path=CONFIG['model_save_path']):
    """Plot training metrics."""
    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(15, 12))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot AUC-ROC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['val_auc'], 'g-')
    plt.title('Validation AUC-ROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.grid(True)

    # Plot Precision and Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['val_precision'], 'c-', label='Precision (Val)')
    plt.plot(epochs, metrics['val_recall'], 'm-', label='Recall (Val)')
    plt.title('Validation Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_metrics.png'))
    plt.close()


# Custom Attention Module (Identical)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        # Apply attention weights
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

# Video Dataset (Added error handling for file existence)
class DeepfakeDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Returning zeros.")
            frames = np.zeros((CONFIG['num_frames'], CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)
        else:
            try:
                frames = extract_frames(video_path, CONFIG['num_frames'])
                if frames.shape[0] != CONFIG['num_frames']:
                     print(f"Warning: extract_frames returned {frames.shape[0]} frames for {video_path}, expected {CONFIG['num_frames']}. Check padding.")
                     # Re-ensure padding (should be handled in extract_frames, but as safeguard)
                     if frames.shape[0] < CONFIG['num_frames']:
                         padding = np.zeros((CONFIG['num_frames'] - frames.shape[0], CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)
                         if frames.shape[0] > 0: # Pad with last frame if possible
                              padding = np.repeat(frames[-1:], CONFIG['num_frames'] - frames.shape[0], axis=0)
                         frames = np.concatenate((frames, padding), axis=0)
                     else: # Truncate if too many (shouldn't happen with linspace)
                         frames = frames[:CONFIG['num_frames']]


            except Exception as e:
                print(f"Error processing video {idx}: {video_path} during frame extraction.")
                print(f"Error details: {str(e)}")
                frames = np.zeros((CONFIG['num_frames'], CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)

        # Apply transformations to each frame
        transformed_frames = []
        if self.transform:
            try:
                for frame_idx, frame in enumerate(frames):
                     # Ensure frame is uint8 for ToPILImage
                     if not isinstance(frame, np.ndarray):
                          print(f"Warning: Frame {frame_idx} for video {video_path} is not a numpy array (type: {type(frame)}). Using zero frame.")
                          frame = np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)
                     elif frame.dtype != np.uint8:
                         # print(f"Warning: Frame {frame_idx} for video {video_path} has dtype {frame.dtype}. Converting to uint8.")
                         frame = frame.astype(np.uint8)

                     # Basic check for valid image dimensions
                     if frame.shape != (CONFIG['frame_size'], CONFIG['frame_size'], 3):
                         print(f"Warning: Frame {frame_idx} for video {video_path} has unexpected shape {frame.shape}. Attempting resize or using zero frame.")
                         try:
                             frame = cv2.resize(frame, (CONFIG['frame_size'], CONFIG['frame_size']))
                             if frame.shape != (CONFIG['frame_size'], CONFIG['frame_size'], 3): # Check resize result
                                 raise ValueError("Resize did not produce expected shape")
                         except Exception as resize_e:
                             print(f"  Resize failed: {resize_e}. Using zero frame.")
                             frame = np.zeros((CONFIG['frame_size'], CONFIG['frame_size'], 3), dtype=np.uint8)


                     transformed_frames.append(self.transform(frame))

                frames_tensor = torch.stack(transformed_frames)
            except Exception as e_transform:
                 print(f"Error applying transform to frames from video {idx}: {video_path}")
                 print(f"Error details: {str(e_transform)}")
                 # Return a dummy tensor in case of transformation errors
                 frames_tensor = torch.zeros((CONFIG['num_frames'], 3, CONFIG['frame_size'], CONFIG['frame_size']))

        else: # If no transform provided
            try:
                # Convert numpy array to tensor: (N, H, W, C) -> (N, C, H, W)
                frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            except Exception as e_notransform:
                print(f"Error converting numpy frames to tensor for video {idx}: {video_path}")
                print(f"Error details: {str(e_notransform)}")
                frames_tensor = torch.zeros((CONFIG['num_frames'], 3, CONFIG['frame_size'], CONFIG['frame_size']))


        # Final check for tensor shape consistency
        if frames_tensor.shape != (CONFIG['num_frames'], 3, CONFIG['frame_size'], CONFIG['frame_size']):
            print(f"Warning: Final tensor shape for video {idx} ({video_path}) is {frames_tensor.shape}, expected {(CONFIG['num_frames'], 3, CONFIG['frame_size'], CONFIG['frame_size'])}. Returning zeros.")
            frames_tensor = torch.zeros((CONFIG['num_frames'], 3, CONFIG['frame_size'], CONFIG['frame_size']))

        return frames_tensor, torch.tensor(label, dtype=torch.float) # Ensure label is float for BCE/Focal loss


# Focal Loss (Identical)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Ensure targets are same type and shape as inputs
        targets = targets.type_as(inputs).view_as(inputs)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # calculates p_t = sigmoid(input) for positive targets and 1 - sigmoid(input) for negative targets
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets) # alpha_t
        focal_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# DeepFake Detection Model (Modified for EfficientNet Freezing)
class DeepfakeDetector(nn.Module):
    def __init__(self, config):
        super(DeepfakeDetector, self).__init__()
        self.config = config

        # Load EfficientNet-B0 pre-trained on ImageNet
        efficientnet = models.efficientnet_b0(weights='DEFAULT')

        # --- MODIFIED FREEZING STRATEGY for EfficientNet ---
        # EfficientNet structure: features (Sequential), avgpool, classifier (Linear)
        # We will freeze initial blocks within 'features'
        feature_blocks = efficientnet.features
        layers_to_freeze = config.get('layers_to_freeze', 3) # Default to 3 if not in config
        print(f"Attempting to freeze the first {layers_to_freeze} feature blocks of EfficientNet...")

        ct = 0
        # Iterate through the direct children of the 'features' Sequential module
        for idx, child in enumerate(feature_blocks.children()):
            if ct < layers_to_freeze:
                print(f"  Freezing EfficientNet feature block {idx}: {child.__class__.__name__}")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(f"  NOT Freezing EfficientNet feature block {idx}: {child.__class__.__name__}")
                for param in child.parameters():
                    param.requires_grad = True # Ensure later layers are trainable
            ct += 1
        print(f"Finished setting requires_grad for EfficientNet feature blocks.")
        # --- END MODIFIED FREEZING ---

        # Use the modified features and the original avgpool, discard original classifier
        self.feature_extractor = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool
        )

        # Feature dimension from EfficientNet-B0 after avgpool
        self.feature_dim = 1280

        # Check feature dimension (optional sanity check)
        # with torch.no_grad():
        #     dummy_input = torch.randn(1, 3, config['frame_size'], config['frame_size'])
        #     dummy_output = self.feature_extractor(dummy_input)
        #     print(f"Sanity Check: Output shape after feature extractor: {dummy_output.shape}") # Should be [1, 1280, 1, 1]
        #     actual_dim = dummy_output.view(dummy_output.size(0), -1).shape[1]
        #     if actual_dim != self.feature_dim:
        #          print(f"Warning: Expected feature_dim {self.feature_dim}, but got {actual_dim}. Adjusting.")
        #          self.feature_dim = actual_dim


        # Spatial dropout (applied after feature extraction, before LSTM)
        # Note: Dropout2d usually applied on feature maps (CxHxW). Here applying on flattened features (effectively 1D dropout)
        self.spatial_dropout = nn.Dropout(p=config['dropout_rate']) if config['use_spatial_dropout'] else nn.Identity()

        # Bidirectional LSTM for temporal analysis
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=config['lstm_hidden_dim'],
            num_layers=2, # Using 2 LSTM layers
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout_rate'] if config['use_temporal_dropout'] and 2 > 1 else 0 # LSTM dropout only applies between layers if num_layers > 1
        )

        # Attention mechanism
        self.use_attention = config['use_attention']
        lstm_output_dim = config['lstm_hidden_dim'] * 2 # *2 for bidirectional

        if self.use_attention:
            self.attention = TemporalAttention(lstm_output_dim)
            # Final classifier using attended features
            self.classifier = nn.Sequential(
                nn.Linear(lstm_output_dim, config['lstm_hidden_dim']), # Input from attention context vector
                nn.ReLU(),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(config['lstm_hidden_dim'], 1)
            )
        else:
            # Final classifier using the last LSTM hidden state (or average/max pooling)
            # Using last hidden state here for simplicity if no attention
             self.classifier = nn.Sequential(
                nn.Linear(lstm_output_dim, config['lstm_hidden_dim']), # Input is last hidden state
                nn.ReLU(),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(config['lstm_hidden_dim'], 1)
            )

    def forward(self, x):
        # x shape: (batch_size, seq_len, C, H, W)
        batch_size, seq_len, c, h, w = x.size()

        # Process each frame through CNN
        # Reshape input for CNN: (batch_size * seq_len, C, H, W)
        x_cnn = x.view(batch_size * seq_len, c, h, w)
        cnn_out = self.feature_extractor(x_cnn) # Output shape: (batch*seq_len, feature_dim, 1, 1)

        # Flatten the features
        cnn_features = cnn_out.view(batch_size, seq_len, -1) # (batch, seq_len, feature_dim)

        # Apply spatial dropout to the sequence of features
        if self.config['use_spatial_dropout']:
             cnn_features = self.spatial_dropout(cnn_features) # Apply dropout across the feature dimension for each time step

        # Process sequence with LSTM
        lstm_out, _ = self.lstm(cnn_features)  # lstm_out shape: (batch, seq_len, hidden_dim*2)

        # Apply attention or use final state
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out) # context shape: (batch, hidden_dim*2)
            # We could potentially return attn_weights for visualization/analysis
        else:
            # Use the output of the last time step from the BiLSTM
            context = lstm_out[:, -1, :] # context shape: (batch, hidden_dim*2)

        # Final classification
        output = self.classifier(context) # output shape: (batch, 1)
        return output.squeeze(-1) # Squeeze the last dimension -> (batch)


# Training function (Identical, but added save paths to plots)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, config):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping_counter = 0
    save_path = config['model_save_path'] # Get save path from config

    # Create scaler for mixed precision training
    scaler = GradScaler()

    # Store metrics
    metrics = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_auc': []
    }

    # Create tqdm progress bar for epochs
    epoch_loop = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")

    for epoch in epoch_loop:
        epoch_loop.set_description(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")
        for inputs, labels in train_loop:
            inputs = inputs.to(device, non_blocking=True) # Use non_blocking for potential speedup
            labels = labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad(set_to_none=True) # More efficient potentially

            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward and optimize with gradient scaling
            scaler.scale(loss).backward()
            # Optional: Gradient clipping (can help stabilize training)
            # scaler.unscale_(optimizer) # Unscale gradients before clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs.detach()) >= 0.5 # Use detach() for metrics calculation
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            # Update progress bar postfix
            train_loop.set_postfix(loss=loss.item(), acc=correct_train/total_train if total_train > 0 else 0)

        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = correct_train / total_train if total_train > 0 else 0
        metrics['train_loss'].append(epoch_train_loss)
        metrics['train_acc'].append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss_val = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        all_outputs = [] # For AUC

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False, unit="batch")
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Forward pass (no autocast needed for eval unless specific layers require it)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Calculate loss for monitoring

                # Statistics
                running_loss_val += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                preds = probs >= 0.5
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                # Store predictions and labels for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(probs.cpu().numpy()) # Store probabilities for AUC

                # Update progress bar postfix
                val_loop.set_postfix(loss=loss.item(), acc=correct_val/total_val if total_val > 0 else 0)

        epoch_val_loss = running_loss_val / total_val if total_val > 0 else 0
        epoch_val_acc = correct_val / total_val if total_val > 0 else 0

        # Calculate additional metrics (handle potential division by zero)
        try:
            val_precision = precision_score(all_labels, all_preds, zero_division=0)
        except ValueError: # Handle cases where labels/preds might be all one class temporarily
             val_precision = 0.0
        try:
            val_recall = recall_score(all_labels, all_preds, zero_division=0)
        except ValueError:
            val_recall = 0.0
        try:
            # Ensure there are samples of both classes for AUC calculation
            if len(np.unique(all_labels)) > 1:
                 val_auc = roc_auc_score(all_labels, all_outputs)
            else:
                 print(f"Warning: Only one class present in validation labels for epoch {epoch+1}. AUC set to 0.5.")
                 val_auc = 0.5 # Or handle as undefined/skip
        except ValueError as e_auc:
             print(f"Could not calculate AUC for epoch {epoch+1}: {e_auc}. Setting to 0.5.")
             val_auc = 0.5


        # Generate confusion matrix for this epoch
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(cm, epoch, save_path=save_path) # Pass save_path

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")
        print(f"  Precision:  {val_precision:.4f} | Recall:    {val_recall:.4f} | AUC: {val_auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print("-" * 60)

        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        # Save metrics
        metrics['val_loss'].append(epoch_val_loss)
        metrics['val_acc'].append(epoch_val_acc)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)
        metrics['val_auc'].append(val_auc)

        # Save model if it's the best so far based on validation accuracy
        if epoch_val_acc > best_val_acc:
            print(f"Validation accuracy improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving model...")
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss # Also save best loss associated with best accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': best_val_loss,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_auc': val_auc,
                'config': config # Save config with the model
            }, os.path.join(save_path, "best_model.pth"))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Validation accuracy did not improve. Counter: {early_stopping_counter}/{config['early_stopping_patience']}")


        # Save checkpoint periodically (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0:
             print(f"Saving checkpoint at epoch {epoch+1}...")
             torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'val_acc': epoch_val_acc,
                 'val_loss': epoch_val_loss,
                 'config': config
             }, os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth"))

        # Early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Plot overall training metrics at the end
    plot_metrics(metrics, save_path=save_path) # Pass save_path

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    # Load best model weights back into the model
    model.load_state_dict(best_model_wts)
    return model, metrics


# Load and process data (Identical, but added check for empty lists)
def prepare_dataset(config):
    """Process the dataset and create data loaders."""
    print("Preparing dataset...")
    data_root = config['data_root']

    # Parse the list file (assuming Celeb-DF v2 structure)
    # Training data comes from Celeb-real, YouTube-real, Celeb-synthesis
    # Test data is defined in List_of_testing_videos.txt (we exclude these from training/validation)
    test_list_path = os.path.join(data_root, 'List_of_testing_videos.txt')
    video_paths = []
    labels = []

    # Read the test list to exclude these videos
    test_videos_relative_paths = set()
    if os.path.exists(test_list_path):
        try:
            with open(test_list_path, 'r') as f:
                # Skip header line if present (check first line format)
                first_line = f.readline().strip()
                # Simple check if it looks like a header vs data
                if not (len(first_line.split()) >= 2 and first_line.split()[0].isdigit()):
                     print("Skipping potential header line in test list.")
                else:
                     # Process the first line if it's data
                     parts = first_line.split()
                     if len(parts) >= 2:
                         # Label seems to be 1 for fake, 0 for real in test list? Double check dataset spec.
                         # Assuming format: label relative_path (e.g., 1 Celeb-synthesis/id0_id1_0000.mp4)
                         test_videos_relative_paths.add(parts[1])


                # Process rest of the file
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        test_videos_relative_paths.add(parts[1])
            print(f"Found {len(test_videos_relative_paths)} test videos to exclude from training/validation.")
        except Exception as e:
             print(f"Error reading test list file {test_list_path}: {e}. Proceeding without exclusions.")
             test_videos_relative_paths = set()

    else:
        print(f"Warning: Test list file not found at {test_list_path}. Cannot exclude test videos.")


    # Collect all video paths from training-related directories
    real_dirs = ['Celeb-real', 'YouTube-real']
    fake_dirs = ['Celeb-synthesis']
    all_video_files = [] # Store tuples of (full_path, label)

    # Process real videos
    for dir_name in real_dirs:
        dir_path = os.path.join(data_root, dir_name)
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory {dir_path} not found. Skipping...")
            continue
        print(f"Processing directory: {dir_path}")
        count = 0
        for file_name in os.listdir(dir_path):
            if file_name.lower().endswith('.mp4'):
                full_path = os.path.join(dir_path, file_name)
                relative_path = os.path.join(dir_name, file_name) # Path relative to data_root used in test list

                # Skip if this video is in the test set
                if relative_path in test_videos_relative_paths:
                    # print(f"Skipping test video: {relative_path}")
                    continue

                all_video_files.append((full_path, 1)) # Real = 1
                count += 1
        print(f"  Found {count} real videos (excluding test set).")


    # Process fake videos
    for dir_name in fake_dirs:
        dir_path = os.path.join(data_root, dir_name)
        if not os.path.isdir(dir_path):
            print(f"Warning: Directory {dir_path} not found. Skipping...")
            continue
        print(f"Processing directory: {dir_path}")
        count = 0
        for file_name in os.listdir(dir_path):
            if file_name.lower().endswith('.mp4'):
                full_path = os.path.join(dir_path, file_name)
                relative_path = os.path.join(dir_name, file_name)

                # Skip if this video is in the test set
                if relative_path in test_videos_relative_paths:
                    # print(f"Skipping test video: {relative_path}")
                    continue

                all_video_files.append((full_path, 0)) # Fake = 0
                count += 1
        print(f"  Found {count} fake videos (excluding test set).")

    if not all_video_files:
         raise ValueError(f"No video files found in specified directories ({real_dirs}, {fake_dirs}) under {data_root} or all were excluded as test videos.")

    # Separate paths and labels
    video_paths = [item[0] for item in all_video_files]
    labels = np.array([item[1] for item in all_video_files])

    # Print class distribution
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 0:
        class_distribution = dict(zip(unique, counts))
        print(f"Class distribution before split (0=Fake, 1=Real): {class_distribution}")
        # Check for imbalance
        if 0 not in class_distribution or 1 not in class_distribution:
             print("Warning: Only one class found in the collected data.")
        elif counts[0] == 0 or counts[1] == 0:
             print("Warning: One class has zero samples.")

    else:
         raise ValueError("No labels collected. Cannot proceed.")


    # Split data into train and validation sets
    try:
         train_paths, val_paths, train_labels, val_labels = train_test_split(
             video_paths, labels,
             test_size=0.2, # 20% for validation
             random_state=42,
             stratify=labels # Important for imbalanced datasets
         )
    except ValueError as e_split:
         if "n_splits=2 cannot be greater than the number of members in each class" in str(e_split):
              print("Error during train/test split: Not enough samples in one class for stratification. Check data collection and class distribution.")
              raise
         else:
              print(f"An unexpected error occurred during train/test split: {e_split}")
              raise


    print(f"Train set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    print(f"Train class distribution (0=Fake, 1=Real): {dict(zip(unique_train, counts_train))}")
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    print(f"Validation class distribution (0=Fake, 1=Real): {dict(zip(unique_val, counts_val))}")


    # Create transformations (Standard for ImageNet pre-trained models)
    # Normalization values for models pre-trained on ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ToPILImage(), # Convert numpy array (H, W, C) to PIL Image
        transforms.Resize((config['frame_size'], config['frame_size'])),
        transforms.RandomHorizontalFlip(p=0.5), # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), # Data augmentation
        # transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), # More augmentation (optional)
        transforms.ToTensor(), # Convert PIL Image to tensor (C, H, W) and scales pixels to [0, 1]
        normalize # Normalize tensor values
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['frame_size'], config['frame_size'])),
        transforms.ToTensor(),
        normalize
    ])

    # Create datasets
    train_dataset = DeepfakeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, transform=val_transform)

    # Calculate class weights for imbalanced data using ONLY training labels
    class_weights = get_class_weights(train_labels)
    print(f"Calculated class weights for WeightedRandomSampler (Fake=0, Real=1): {class_weights.tolist()}")

    # Create weighted sampler for training data to handle imbalance
    # Weight for each sample is the inverse frequency of its class
    sample_weights = torch.tensor([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights), # Draw as many samples as the original dataset size
        replacement=True # Allow drawing same sample multiple times within an epoch
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler, # Use the weighted sampler for training
        num_workers=config['num_workers'],
        pin_memory=True, # Speeds up data transfer to GPU
        prefetch_factor=2 if config['num_workers'] > 0 else None, # How many batches to prefetch
        persistent_workers=True if config['num_workers'] > 0 else False, # Keep workers alive between epochs
        # collate_fn=None, # Default collate_fn is usually fine
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'], # Can often use a larger batch size for validation if memory allows
        shuffle=False, # No need to shuffle validation data
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=2 if config['num_workers'] > 0 else None,
        persistent_workers=True if config['num_workers'] > 0 else False,
    )

    print(f"Data loaders created.")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, class_weights # Returning class_weights maybe useful later if needed for loss


def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    try:
        # Create model save directory (using path from config)
        os.makedirs(CONFIG['model_save_path'], exist_ok=True)
        print(f"Model checkpoints will be saved to: {CONFIG['model_save_path']}")

        # Prepare data
        try:
            train_loader, val_loader, class_weights = prepare_dataset(CONFIG)
        except Exception as e_data:
            print(f"FATAL: Error preparing dataset: {str(e_data)}")
            import traceback
            traceback.print_exc()
            # Optionally print directory structure for debugging data path issues
            print("\n--- Directory Structure ---")
            try:
                for root, dirs, files in os.walk(CONFIG['data_root'], topdown=True):
                     level = root.replace(CONFIG['data_root'], '').count(os.sep)
                     indent = ' ' * 4 * (level)
                     print(f'{indent}{os.path.basename(root)}/')
                     subindent = ' ' * 4 * (level + 1)
                     # Limit files shown per directory
                     files_shown = 0
                     max_files_to_show = 3
                     for f in files:
                          if files_shown < max_files_to_show:
                              print(f'{subindent}{f}')
                              files_shown += 1
                          else:
                              print(f'{subindent}[... and {len(files) - max_files_to_show} more files]')
                              break
                     # Prune deeper exploration if needed for brevity
                     # if level >= 1:
                     #      dirs[:] = [] # Don't go deeper than level 1
            except Exception as e_walk:
                print(f"Could not print directory structure: {e_walk}")
            print("--- End Directory Structure ---\n")

            return # Stop execution if data prep fails


        # Initialize model
        print("Initializing model...")
        model = DeepfakeDetector(CONFIG).to(device)

        # Count trainable parameters (after freezing)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


        # Define loss function
        if CONFIG['use_focal_loss']:
            # Note: Class weights from sampler handle imbalance at data level.
            # Focal loss handles imbalance at loss level (focusing on hard examples).
            # Using both can be effective. We don't pass class weights directly to FocalLoss here.
            criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])
            print(f"Using Focal Loss (alpha={CONFIG['focal_alpha']}, gamma={CONFIG['focal_gamma']})")
        else:
            # If not using Focal Loss, consider using pos_weight in BCEWithLogitsLoss for imbalance
            # pos_weight = weight for the positive class (Real=1)
            # If class_weights = [weight_fake, weight_real], pos_weight = weight_real / weight_fake
            # pos_weight_val = class_weights[1] / class_weights[0] if class_weights[0] > 0 else 1.0
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))
            # print(f"Using BCE Loss with pos_weight={pos_weight_val:.2f}")
             criterion = nn.BCEWithLogitsLoss() # Simpler BCE without explicit weighting if sampler is used
             print("Using standard BCE Loss (relying on WeightedRandomSampler for imbalance)")


        # Define optimizer
        optimizer = optim.AdamW(
            model.parameters(), # Pass all parameters; optimizer respects requires_grad=False
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )
        print(f"Using AdamW optimizer (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
        # --- Consider adjusting LR if performance is poor ---
        print("--> NOTE: If initial training is unstable or accuracy is very low, consider lowering the learning rate (e.g., 1e-5 or 5e-5) for EfficientNet.")


        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', # Reduce LR when validation loss stops decreasing
            factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience'],
            verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler (factor={CONFIG['scheduler_factor']}, patience={CONFIG['scheduler_patience']})")

        # Train the model
        print("\n--- Starting Training ---")
        model, metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=CONFIG['num_epochs'],
            config=CONFIG # Pass config for saving paths etc.
        )

        print("\n--- Training Completed ---")

        # Save the final model (which is the best model based on validation accuracy)
        # The best model state is already loaded back into 'model' by train_model
        final_model_path = os.path.join(CONFIG['model_save_path'], "final_best_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(), # Contains the best weights
            'optimizer_state_dict': optimizer.state_dict(), # State at the end of training
            'scheduler_state_dict': scheduler.state_dict(), # State at the end of training
            'config': CONFIG,
            'metrics': metrics # Optionally save training history
        }, final_model_path)

        print(f"Best model (based on validation accuracy) saved to {final_model_path}")

    except Exception as e:
        print(f"\n--- An error occurred during the main execution ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()