# finetune method and related class

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
import torchvision
import copy

import dataset_pre

# Optimizer
MOMENTUM = 0.9
STEP=5
GAMMA=0.5

# Size
BATCH_SIZE = 32
EPOCH = 50

num_features_in = 4096
num_features_out = 1000

# Adjust to fit vgg16
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# finetune main process
def train_model(model, device, criterion, optimizer, scheduler, train_set, train_dataloader, test_set, test_dataloader, num_epochs):

    for epoch in range(num_epochs):
        # Train
        model.train()
        torch.cuda.empty_cache()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        train_loss = 0.0

        # Iterate over data.
        for batch_idx, (image, target) in enumerate(train_dataloader):
            image = image.to(device)
            target = target.to(device)

            with torch.set_grad_enabled(True):
                outputs = model(image)
                loss = criterion(outputs, target)

            train_loss += loss.item() * image.size(0)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss = train_loss / len(train_set)
        
        print(f'Epoch: {epoch+1}/{num_epochs} Train Loss: {epoch_loss:.4f}')
        scheduler.step()
        
        # Test
        model.eval()
        print('Begin test......')
        features = []
        targets = []
        
        # Variables to store test loss and correct predictions
        test_loss = 0.0
        correct_predictions = 0

        # Iterate over the test dataset
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                features.append(outputs.view(outputs.size(0), -1).cpu())
                targets.append(labels.cpu())

                loss = criterion(outputs, labels)

                # Update test loss
                test_loss += loss.item() * inputs.size(0)

                # Calculate the number of correct predictions
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)

        # Calculate average test loss
        average_loss = test_loss / len(test_set)

        # Calculate test accuracy
        accuracy = correct_predictions.double() / len(test_set)

        # Print the results
        print(f"Test Loss: {average_loss:.4f},Test Accuracy: {accuracy:.4f}")
    
    return model, accuracy, torch.cat(features), torch.cat(targets)

class MeanCentroidClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MeanCentroidClassifier, self).__init__()
        self.centroids = nn.Parameter(torch.randn(num_classes, input_dim), requires_grad=True)

    def forward(self, x):
        # x: (batch_size, input_dim)
        # Calculate negative cosine similarities
        similarities = -F.cosine_similarity(x.unsqueeze(1), self.centroids.unsqueeze(0), dim=2)
        # Return logits
        return similarities

class CosineSimilarityClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CosineSimilarityClassifier, self).__init__()
        self.weights = nn.Parameter(torch.randn(num_classes, input_dim))

    def forward(self, x):
        # x: (batch_size, input_dim)
        # Normalize input and weights
        x_normalized = F.normalize(x, p=2, dim=1)
        weights_normalized = F.normalize(self.weights, p=2, dim=1)
        # Calculate cosine similarities
        similarities = F.linear(x_normalized, weights_normalized)
        # Return logits
        return similarities

# Fixed feature extracter template
def fixed_feature_extracter(model, num_features_out, total_classes_num, lr):
    # Add a new layer
    model.classifier.add_module('7', nn.Linear(num_features_out, total_classes_num))
    
    # Only finetune the added layer
    optimizer = optim.SGD(model.classifier[7].parameters(), lr=lr, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    
    return model, optimizer, scheduler

# All fit template
def all_fit(mode, model, num_features_in, total_classes_num, lr):
    
    # Change model according to mode
    if mode == "linear":
        model.classifier[-1] = nn.Linear(num_features_in, total_classes_num)
    elif mode == "mean_centroid":
        model.classifier[-1] = MeanCentroidClassifier(num_features_in, total_classes_num)
    elif mode == "cosine":
        model.classifier[-1] = CosineSimilarityClassifier(num_features_in, total_classes_num)
    
    # Adjust optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    
    return model, optimizer, scheduler

# Fit last k template
def fit_last(model, num_features_in, total_classes_num, lr, k):
    # k = 1,2,3. Only finetune last k layers
    model.classifier[-1] = nn.Linear(num_features_in, total_classes_num)
    
    # Select the parameters to update based on k
    optimizer = optim.SGD(model.classifier[-(3*k-2)].parameters(), lr=lr, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
    
    return model, optimizer, scheduler

# k-shot for 5 times
def k_shot(base_folder, shot_folder, test_folder, images_per_class, num_total_classes, lr, criterion, device, mode):
    accs = [] 
    for i in range(10):
        dataset_pre.split_shot_data(base_folder, base_folder+shot_folder, base_folder+test_folder, num_total_classes, images_per_class)
        shot_set = torchvision.datasets.ImageFolder(root=base_folder+shot_folder,transform=data_transforms)
        shot_dataloader = torch.utils.data.DataLoader(dataset=shot_set,batch_size=BATCH_SIZE,shuffle=True)
        test_set = torchvision.datasets.ImageFolder(base_folder+test_folder,transform=data_transforms)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=True)
        
        if mode == "baseline":
            model = models.vgg16(pretrained=False)
            model, optimizer, scheduler = all_fit("linear", model, num_features_in, num_total_classes, lr)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "extract_feature":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = fixed_feature_extracter(model, num_features_out, num_total_classes, lr)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "af_linear":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = all_fit("linear", model, num_features_in, num_total_classes, lr)
            model = model.to(device)
            if i == 9:
                model_before_finetune = copy.deepcopy(model)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "af_centroid":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = all_fit("mean_centroid", model, num_features_in, num_total_classes, lr)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "af_cosine":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = all_fit("cosine", model, num_features_in, num_total_classes, lr)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "fit_l1":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = fit_last(model, num_features_in, num_total_classes, lr, 1)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "fit_l2":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = fit_last(model, num_features_in, num_total_classes, lr, 2)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        elif mode == "fit_l3":
            model = models.vgg16(pretrained=True)
            model, optimizer, scheduler = fit_last(model, num_features_in, num_total_classes, lr, 3)
            model = model.to(device)
            after_model, accuracy, features, labels = train_model(model, device, criterion, optimizer, scheduler, 
                                                 shot_set, shot_dataloader, test_set, test_dataloader, EPOCH)
        else:
            print("error")
        accs.append(accuracy.item())
    if mode == "af_linear":
        return accs, features, labels, model_before_finetune, after_model
    return accs, features, labels
        
# Parameters comparison
def compare_params(model_before, model_after):
    # Store the change rate of each layer's parameters
    param_changes = {}

    for name_before, param_before in model_before.named_parameters():
        if name_before == "classifier.6.weight" or name_before == "classifier.6.bias":
            continue
        # Parameters of the fine-tuning model
        param_after = dict(model_after.named_parameters())[name_before]

        # The rate of change of parameters
        change_percentage = torch.norm(param_after - param_before) / torch.norm(param_before)
        param_changes[name_before] = change_percentage.item()

    return param_changes