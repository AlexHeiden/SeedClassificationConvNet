import os
import cv2
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report


class CNNNet(nn.Module):

    def __init__(self, num_classes=3):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 16))
        self.classifier = nn.Sequential(
            nn.Linear(8192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=4, device="cpu"):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

    for epoch in range(1, epochs + 1):
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        model.train()
        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            output = model(inputs)

            #calculation of intermediate results and loss function values
            train_loss = loss_fn(output, targets)
            train_acc = multi_acc(output, targets)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                output = model(inputs)

                val_loss = loss_fn(output, targets)
                val_acc = multi_acc(output, targets)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        scheduler.step()

        #writing intermediate results into dictionaries
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        print(
            f'Epoch {epoch + 0:03}: | Train Loss: '
            f'{train_epoch_loss / len(train_loader):.5f} | Val Loss: '
            f'{val_epoch_loss / len(val_loader):.5f} | Train Acc: '
            f'{train_epoch_acc / len(train_loader):.5f}| Val Acc: '
            f'{val_epoch_acc / len(val_loader):.5f}')


#returns the distribution of objects between categories to distribute the weights in train_data
def get_class_distribution(y):
    count_dict = {
        "rating_barley": 0,
        "rating_rye": 0,
        "rating_wheat": 0
    }

    for i in y:
        if i == 0:
            count_dict['rating_barley'] += 1
        elif i == 1:
            count_dict['rating_rye'] += 1
        elif i == 2:
            count_dict['rating_wheat'] += 1
        else:
            print('Check classes')

    return count_dict


#returns accuracy of classification
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


#returns y_pred and y_test of a specific category
#writes incorrectly classified objects into special mistakes_folders
def get_accuracy(data_loader, data, dst, output_mistakes=False):
    y_pred_list = []
    with torch.no_grad():
        cnnnet.eval()
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            y_test_pred = cnnnet(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    y_test = []
    for _, t in data:
        y_test.append(t)
    y_test = list(y_test)

    if (output_mistakes):
        imgs = data_loader.dataset.imgs

        #deleting all files from Mistakes_folders
        for filename in os.listdir(dst + 'barley/'):
            os.remove(dst + 'barley/' + filename)

        for filename in os.listdir(dst + 'barley-rye/'):
            os.remove(dst + 'barley-rye/' + filename)

        for filename in os.listdir(dst + 'barley-wheat/'):
            os.remove(dst + 'barley-wheat/' + filename)

        for filename in os.listdir(dst + 'rye/'):
            os.remove(dst + 'rye/' + filename)

        for filename in os.listdir(dst + 'rye-barley/'):
            os.remove(dst + 'rye-barley/' + filename)

        for filename in os.listdir(dst + 'rye-wheat/'):
            os.remove(dst + 'rye-wheat/' + filename)

        for filename in os.listdir(dst + 'wheat/'):
            os.remove(dst + 'wheat/' + filename)

        for filename in os.listdir(dst + 'wheat-barley/'):
            os.remove(dst + 'wheat-barley/' + filename)

        for filename in os.listdir(dst + 'wheat-rye/'):
            os.remove(dst + 'wheat-rye/' + filename)

        #filling special mistakes_folders with incorrectly classified objects
        for i in range(len(y_test)):
            if y_test[i] == 0:
                if y_pred_list[i] == 0:
                    cv2.imwrite(dst + 'barley/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
                elif y_pred_list[i] == 1:
                    cv2.imwrite(dst + 'barley-rye/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
                elif y_pred_list[i] == 2:
                    cv2.imwrite(dst + 'barley-wheat/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
            elif y_test[i] == 1:
                if y_pred_list[i] == 0:
                    cv2.imwrite(dst + 'rye-barley/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
                elif y_pred_list[i] == 1:
                    cv2.imwrite(dst + 'rye/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
                elif y_pred_list[i] == 2:
                    cv2.imwrite(dst + 'rye-wheat/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
            elif y_test[i] == 2:
                if y_pred_list[i] == 0:
                    cv2.imwrite(dst + 'wheat-barley/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
                elif y_pred_list[i] == 1:
                    cv2.imwrite(dst + 'wheat-rye/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))
                elif y_pred_list[i] == 2:
                    cv2.imwrite(dst + 'wheat/' + imgs[i][0].split('\\')[1], cv2.imread(imgs[i][0]))

    return y_pred_list, y_test

#dictionaries for intermediate results
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

#preprocessing images
img_transforms = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor()
])

#creating datasets from folders with images
train_data_path = "./data/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms)

val_data_path = "./data/val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms)

test_data_path = "./data/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms)

#distributing categories' weights
# target_list = []
# for _, t in train_data:
#     target_list.append(t)
#
# target_list = torch.tensor(target_list)
# class_count = [i for i in get_class_distribution(target_list).values()]
# class_weights = 1./torch.tensor(class_count, dtype=torch.float)
# class_weights_all = class_weights[target_list]
# weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)

#uploading images into DataLoaders and splitting them into minibatches
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

#training the neural network
device = 'cpu'
# cnnnet = CNNNet()
# cnnnet.train()
# cnnnet.to(device)
# optimizer = optim.Adam(cnnnet.parameters(), lr=0.0005)
# train(cnnnet, optimizer, torch.nn.CrossEntropyLoss(weight=class_weights.to(device)), train_data_loader,val_data_loader, epochs=35, device='cpu')
# torch.save(cnnnet, './best_seed_classification_cnnnet')

#displaying line charts of intermediate accuracy values and loss function values
# train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
# sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
# sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
# plt.waitforbuttonpress()
# plt.close(fig)

cnnnet = torch.load('./best_seed_classification_cnnnet')
cnnnet.eval()

#receiving the classification accuracy of test_data
y_pred_list, y_test = get_accuracy(test_data_loader, test_data, './err/test/', output_mistakes=True)

class2idx = {
    'barley':0,
    'rye':1,
    'wheat':2
}

idx2class = {v: k for k, v in class2idx.items()}

#calculating the quality of classification and creating a confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
sns.heatmap(confusion_matrix_df, annot=True)
print(classification_report(y_test, y_pred_list, digits=5))
plt.waitforbuttonpress()
