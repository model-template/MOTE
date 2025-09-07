import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data_loader import load_data_training
from .mote import FaceTemplateClassifier1, FaceTemplateClassifier2, FaceTemplateClassifierSmall, RandomFaceTemplateClassifier
from .kde_high_dimension import sample_from_kde


def create_train_val_loader(database_embeddings, test_embedding, kde_model_male, kde_model_female, batch_size, male_ratio=0.5):
    length_embeddings = len(database_embeddings)
    imposter_labels = torch.zeros(length_embeddings)

    num_samples_to_generate = length_embeddings // 2
    new_samples_male = sample_from_kde(kde_model_male, test_embedding, n_samples=int(num_samples_to_generate * male_ratio))
    new_samples_female = sample_from_kde(kde_model_female, test_embedding, n_samples=int(num_samples_to_generate * (1 - male_ratio)))

    new_samples = torch.from_numpy(np.concatenate((new_samples_male, new_samples_female), axis=0)).float()
    new_labels = torch.ones(len(new_samples))

    final_embeddings = torch.cat((torch.from_numpy(database_embeddings).float(), new_samples), dim=0)
    final_labels = torch.cat((imposter_labels, new_labels), dim=0)

    dataset = TensorDataset(final_embeddings, final_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_model_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def validate_model_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)

def train_single_mote(model, train_loader, val_loader, criterion, optimizer, scheduler, device, patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.TRAINING_PARAMS['num_epochs']):
        train_loss = train_model_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model_temp.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                #print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    model.load_state_dict(torch.load('best_model_temp.pth'))
    return model

def run_training(architecture, dataset):
    print(f"Starting training for architecture: {architecture}, dataset: {dataset}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_embeddings, train_identities, kde_model_train, kde_model_train_male, kde_model_train_female = load_data_training(architecture, 'morph')
    val_embeddings, val_identities, kde_model_val, kde_model_val_male, kde_model_val_female = load_data_training(architecture, 'lfw')
    test_embeddings, test_identities, _, _, _ = load_data_training(architecture, dataset)

    grouped_embeddings = {identity: test_embeddings[test_identities == identity] for identity in np.unique(test_identities)}

    # Create save directory
    save_dir = os.path.join(config.PRETRAINED_MODELS_DIR, f"mote_{architecture}_{dataset}")
    os.makedirs(save_dir, exist_ok=True)

    # Loop through each identity and train a model
    for identity, embeddings in grouped_embeddings.items():
        print(f"Training MOTE for identity: {identity}")
        model = FaceTemplateClassifier1().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.00005)

        reference_embedding = embeddings[0]

        train_loader = create_train_val_loader(train_embeddings, reference_embedding, kde_model_train_male, kde_model_train_female, config.TRAINING_PARAMS['batch_size'])
        val_loader = create_train_val_loader(val_embeddings, reference_embedding, kde_model_val_male, kde_model_val_female, config.TRAINING_PARAMS['batch_size'])

        scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=config.TRAINING_PARAMS['num_epochs'])

        trained_model = train_single_mote(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config.TRAINING_PARAMS['patience'])

        # Save the trained model
        model_save_path = os.path.join(save_dir, f"model_1_identity_{identity}.pth")
        torch.save(trained_model.state_dict(), model_save_path)

    print(f"Finished training for all identities in {dataset}.")
