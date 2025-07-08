import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple

# --- Global Configuration Variables (CAPITALIZED) ---
XRAY_DATA_PATH: str = './data'
RAW_SUBFOLDERS: List[str] = ['01', '02', '03']
CLASS_NAMES: List[str] = ["Class1", "Class2", "Class3"]
IMAGE_SIZE: Tuple[int, int] = (300, 300)
TRAIN_SPLIT_RATIO: float = 0.6
VAL_SPLIT_RATIO: float = 0.5  # 0.5 of the remaining 0.4 (i.e., 0.2 of total)
TEST_SPLIT_RATIO: float = 0.5  # 0.5 of the remaining 0.4 (i.e., 0.2 of total)
RANDOM_SEED: int = 42

# Training/Optuna Hyperparameters (Adjust for faster testing)
PATIENCE: int = 3 # Reduced for faster testing
NUM_TRIALS: int = 5 # Reduced for faster testing
NUM_EPOCHS_OPTUNA: int = 5 # Reduced max epochs for faster testing

# --- Device Setup ---
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Transformations ---
DATA_TRANSFORMS: Dict[str, transforms.Compose] = {
    'train': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# --- Functions ---

def verify_source_data_existence(data_path: str, subfolders: List[str]) -> None:
    """
    Verifies that the initial data subfolders exist and contain images.
    """
    print(f"Verifying source data in: {data_path}")
    for subfolder in subfolders:
        subfolder_path = os.path.join(data_path, subfolder)
        if not os.path.exists(subfolder_path):
            raise FileNotFoundError(f"Subfolder {subfolder_path} not found.")
        images = [f for f in os.listdir(subfolder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        print(f"  {subfolder}: {len(images)} images found")
        if not images:
            raise FileNotFoundError(f"No images found in {subfolder_path}.")
    print("Source data verification complete!")


def prepare_data_splits(base_data_path: str, raw_subfolders: List[str], class_names_map: List[str],
                        train_ratio: float, val_ratio: float, test_ratio: float, random_state: int
                        ) -> Tuple[str, str, str]:
    """
    Splits the raw image data into train, validation, and test sets,
    creating new directory structures and copying images.
    """
    print("\nPreparing data splits...")
    train_dir, val_dir, test_dir = _initialize_split_directories(base_data_path, class_names_map)

    for old_class, new_class in zip(raw_subfolders, class_names_map):
        source_class_path = os.path.join(base_data_path, old_class)
        _split_and_copy_class_images(
            source_class_path, train_dir, val_dir, test_dir, new_class,
            train_ratio, val_ratio, test_ratio, random_state
        )
    
    _verify_split_directory_counts(base_data_path, class_names_map, train_dir, val_dir, test_dir)
    print("Data splitting and copying complete!")
    return train_dir, val_dir, test_dir

# --- Helper functions (would be defined below prepare_data_splits or in a separate module) ---

def _initialize_split_directories(base_data_path: str, class_names_map: List[str]) -> Tuple[str, str, str]:
    """Calculates, cleans, and creates the train, val, and test directories with class subfolders."""
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_dir = os.path.join(base_data_path, "test")

    for dir_path in [train_dir, val_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        for class_name in class_names_map:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
    print("  Cleaned and created train, val, test directories.")
    return train_dir, val_dir, test_dir

def _split_and_copy_class_images(source_class_path: str, dest_train_dir: str, dest_val_dir: str,
                                dest_test_dir: str, new_class_name: str, train_ratio: float,
                                val_ratio: float, test_ratio: float, random_state: int ) -> None:
    
    """Reads images from a source class, splits them, and copies to target directories."""
    
    images = [f for f in os.listdir(source_class_path) if f.endswith((".jpg", ".png", ".jpeg"))]
    image_paths = [os.path.join(source_class_path, img) for img in images]

    train_imgs, temp_imgs = train_test_split(image_paths, test_size=(1 - train_ratio), 
                                             random_state=random_state)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), 
                                           random_state=random_state)

    for img_path in train_imgs:
        shutil.copy(img_path, os.path.join(dest_train_dir, new_class_name, os.path.basename(img_path)))
    for img_path in val_imgs:
        shutil.copy(img_path, os.path.join(dest_val_dir, new_class_name, os.path.basename(img_path)))
    for img_path in test_imgs:
        shutil.copy(img_path, os.path.join(dest_test_dir, new_class_name, os.path.basename(img_path)))
    
    print(f"  Class {new_class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")


def _verify_split_directory_counts(base_data_path: str, class_names_map: List[str], train_dir: str, 
                                   val_dir: str, test_dir: str) -> None:
    
    """Verifies that the correct number of files exist in the split directories."""

    for dir_path, split_name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        for class_name in class_names_map:
            class_dir = os.path.join(dir_path, class_name)
            num_files = len([f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
            print(f"  {split_name}/{class_name}: {num_files} files")
            if num_files == 0:
                raise RuntimeError(f"No files found in {class_dir}. Data splitting failed.")
    print("  Split directory verification complete.")


def load_datasets(train_dir: str, val_dir: str, test_dir: str, transforms_dict: Dict[str, transforms.Compose]
                    ) -> Tuple[ImageFolder, ImageFolder, ImageFolder]:
    """
    Loads image datasets using ImageFolder from specified directories.
    """

    print("\nLoading datasets...")
    train_dataset = ImageFolder(root=train_dir, transform=transforms_dict['train'])
    val_dataset = ImageFolder(root=val_dir, transform=transforms_dict['val'])
    test_dataset = ImageFolder(root=test_dir, transform=transforms_dict['test'])

    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size: {len(val_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    print("  Class mapping:", train_dataset.class_to_idx)
    print("Datasets loaded!")
    return train_dataset, val_dataset, test_dataset

def create_model(num_classes: int, dropout_rate: float = 0.35) -> nn.Module:
    """
    Creates an EfficientNet-B3 model with a custom classification head.
    """

    print(f"Creating EfficientNet-B3 model with {num_classes} classes and dropout {dropout_rate}...")
    model = EfficientNet.from_pretrained('efficientnet-b3')
    num_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, num_classes)
    )
    return model.to(DEVICE)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, num_epochs: int, patience: int = 5 ) -> float:
    """
    Trains the given model with early stopping and learning rate scheduling.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.

    Returns:
        float: The best validation accuracy achieved during training.
    """
    print(f"\nStarting model training for {num_epochs} epochs with patience {patience}...")
    best_val_acc = 0.0
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)  # Adjust LR based on val_acc
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_efficientnet.pth")
            patience_counter = 0
            print("  New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} due to no improvement in validation accuracy.")
                break
    
    print("Model training complete.")
    return best_val_acc


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    """
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.3, 0.5)
    
    # Use the global constant for max epochs for Optuna trials
    num_epochs = trial.suggest_int("num_epochs", 2, NUM_EPOCHS_OPTUNA) # Min 2 epochs for sanity

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = create_model(num_classes=len(CLASS_NAMES), dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    
    val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=PATIENCE)
    return val_acc

def evaluate_model(model: nn.Module, test_loader: DataLoader, class_names: List[str]) -> None:
    """
    Evaluates the trained model on the test set and prints performance metrics.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        class_names (List[str]): List of class names for interpretation.
    """
    print("\nEvaluating model on the test set...")
    all_preds = []
    all_labels = []
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Model evaluation complete!")

def classify_single_image(model: nn.Module, image_path: str, transforms: transforms.Compose, class_names: List[str]) -> str:
    """
    Classifies a single image using the trained model.

    Args:
        model (nn.Module): The trained PyTorch model.
        image_path (str): Path to the image file to classify.
        transforms (transforms.Compose): Transforms to apply to the single image.
        class_names (List[str]): List of class names for interpretation.

    Returns:
        str: The predicted class name.
    """
    print(f"\nClassifying single image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    input_tensor = transforms(image).unsqueeze(0).to(DEVICE) # Add batch dimension
    
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        predicted_class_name = class_names[class_idx]
    print(f"  Predicted class: {predicted_class_name}")
    return predicted_class_name

def export_model(model: nn.Module, output_path: str = './efficientnet_b3_classifier.pt') -> None:
    """
    Exports the trained model to TorchScript or saves its state dictionary.

    Args:
        model (nn.Module): The trained PyTorch model.
        output_path (str): The desired path for the exported model.
    """
    print("\nAttempting to export model...")
    model.eval() # Set model to evaluation mode before tracing
    example_input = torch.rand(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(DEVICE)
    try:
        traced_model = torch.jit.trace(model, example_input, strict=False)
        torch.jit.save(traced_model, output_path)
        print(f"  Model exported to '{output_path}' as TorchScript.")
    except RuntimeError as e:
        print(f"  Tracing failed: {e}. Saving model state dictionary instead.")
        state_dict_path = output_path.replace('.pt', '_state_dict.pth')
        torch.save(model.state_dict(), state_dict_path)
        print(f"  Model state dictionary saved to '{state_dict_path}'.")


# --- Main Execution Flow ---

if __name__ == "__main__":
    print("Starting X-ray Image Classification Pipeline")

    # 1. Verify Source Data
    verify_source_data_existence(XRAY_DATA_PATH, RAW_SUBFOLDERS)

    # 2. Prepare Data Splits
    train_dir, val_dir, test_dir = prepare_data_splits(
        XRAY_DATA_PATH,
        RAW_SUBFOLDERS,
        CLASS_NAMES,
        TRAIN_SPLIT_RATIO,
        VAL_SPLIT_RATIO,
        TEST_SPLIT_RATIO,
        RANDOM_SEED
    )

    # 3. Load Datasets
    train_dataset, val_dataset, test_dataset = load_datasets(
        train_dir, val_dir, test_dir, DATA_TRANSFORMS
    )

    # 4. Hyperparameter Optimization with Optuna
    print("\nStarting Hyperparameter Optimization with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=NUM_TRIALS)

    print("\nOptuna Optimization Complete!")
    print("Best hyperparameters:", study.best_params)
    print("Best validation accuracy:", study.best_value)

    # 5. Train Final Model with Best Parameters
    print("\nTraining final model with best hyperparameters...")
    best_params = study.best_params
    
    final_train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    final_val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
    final_test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

    final_model = create_model(num_classes=len(CLASS_NAMES), dropout_rate=best_params["dropout_rate"])
    final_criterion = nn.CrossEntropyLoss()
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params["lr"], weight_decay=5e-5)
    
    # Use the best_params' num_epochs for the final training
    train_model(final_model, final_train_loader, final_val_loader, final_criterion, final_optimizer,
                num_epochs=best_params["num_epochs"], patience=PATIENCE)

    # Load the best model state (saved during training)
    final_model.load_state_dict(torch.load("best_efficientnet.pth"))
    
    # 6. Evaluate on Test Set
    evaluate_model(final_model, final_test_loader, CLASS_NAMES)

    # 7. Classify a Single X-ray Example
    # Ensure there's at least one image in Class1 for demonstration
    example_image_folder = os.path.join(test_dir, CLASS_NAMES[0])
    if os.path.exists(example_image_folder) and os.listdir(example_image_folder):
        first_image_in_class1 = os.path.join(example_image_folder, os.listdir(example_image_folder)[0])
        classify_single_image(final_model, first_image_in_class1, DATA_TRANSFORMS['test'], CLASS_NAMES)
    else:
        print(f"\nCould not find example image in {example_image_folder} for single image classification demo.")

    # 8. Export to TorchScript
    export_model(final_model)

    print("\nPipeline execution complete!")