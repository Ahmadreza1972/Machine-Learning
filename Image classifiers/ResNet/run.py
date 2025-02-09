import torch
from tqdm import tqdm
import torch.nn.functional as F
from CustomResNet18 import CustomResNet18
from DataLoad import DataLoad
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot  as plt


def load_class_names(filepath):
    with open(filepath, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

def pic(classes,true_classes, my_predict, data):

    

    true_classes_name = classes[true_classes]

    my_predict_name = classes[my_predict] 
    
    grid_img = make_grid(data[0].unsqueeze(0), nrow=1)  
    plt.imshow(grid_img.permute(1, 2, 0))  
    plt.title(f"Predicted: {my_predict_name}, True: {true_classes_name}")
    plt.axis('off') 
    plt.show()



def get_predictions_and_probabilities(model, orginallabels,dataloader, device='cpu'):
    model.to(device)
    predictions = []
    probabilities = []
    pr_lable=[]
    Tr_lable=[]
    tot=0
    correct=0
    
    with torch.no_grad():  # No need to calculate gradients during inference
        for inputs, labels,orglabel in tqdm(dataloader, desc="Making predictions"):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities using softmax
            probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            
            # Get predicted class (index of the maximum probability)
            _, predicted_classes = torch.max(probs, 1)
            
            probabilities.extend(probs.tolist())
            pr_lable.extend(predicted_classes.tolist()) 
            Tr_lable.extend(orglabel.tolist()) 
            tot+=1
            pr=predicted_classes.tolist()[0]
            if (orginallabels[pr]==orglabel.tolist()[0]):
                correct+=1
    accuracy=correct/tot
    return accuracy, probabilities,pr_lable,Tr_lable



def get_union_data(orginallabels):
    
    
    # first model
    train_path1=r"Image classifiers\ResNet\Model1\model1_test.pth"
    test_path1=r"Image classifiers\ResNet\Model1\model1_test.pth"
    Loader1=DataLoad(train_path1,test_path1,0.2,1)
    train_loader1,val_loader1,test_loader1=Loader1.DataLoad()

    # second model    
    train_path2=r"Image classifiers\ResNet\Model2\model2_test.pth"
    test_path2=r"Image classifiers\ResNet\Model2\model2_test.pth"
    Loader2=DataLoad(train_path2,test_path2,0.2,1)
    train_loader2,val_loader2,test_loader2=Loader2.DataLoad()

    # third model    
    train_path1=r"Image classifiers\ResNet\Model3\model3_test.pth"
    test_path1=r"Image classifiers\ResNet\Model3\model3_test.pth"
    Loader3=DataLoad(train_path1,test_path1,0.2,1)
    train_loader3,val_loader3,test_loader3=Loader3.DataLoad()
    
    combined_data=[]
    combined_labels=[]
    orginal_label=[]
    for (data,label) in test_loader1:
        combined_data.append(data)
        combined_labels.append(label)
        orginal_label.append(orginallabels[0][label])
    
    for (data,label) in test_loader2:
        combined_data.append(data)
        combined_labels.append(label)
        orginal_label.append(orginallabels[1][label])
    
    for (data,label) in test_loader3:
        combined_data.append(data)
        combined_labels.append(label)
        orginal_label.append(orginallabels[2][label])  
        
    dataset = TensorDataset(torch.cat(combined_data, dim=0), torch.cat(combined_labels, dim=0),torch.tensor(orginal_label))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
            
    return dataloader

def main():
    # List of model file paths and corresponding original labels
    model_paths = [
    r"Image classifiers\ResNet\Model1\model_weights1.pth",
    r"Image classifiers\ResNet\Model2\model_weights2.pth",
    r"Image classifiers\ResNet\Model3\model_weights3.pth",
    ]

    # Assuming orginallabels is a list of labels for each model
    results = []
    orginallabels=[[0,10,20,30,40],[1,11,21,31,41],[2,12,22,32,42]]
        
    dataloader=get_union_data(orginallabels)
    
    model = CustomResNet18(num_classes=5, freeze_layers=False)

    # Iterate through each model and evaluate
    for i, model_path in enumerate(model_paths):
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Get predictions, probabilities, etc.
        accuracy, probabilities, pr_label, tr_label = get_predictions_and_probabilities(
            model, orginallabels[i], dataloader, device='cuda'
        )

        # Store results in a list or dictionary for later use
        results.append({
            "model": f"Model {i+1}",
            "accuracy": accuracy,
            "probabilities": probabilities,
            "predicted_labels": pr_label,
            "true_labels": tr_label
        })

        # Print the accuracy
        print(f"Model {i+1} Accuracy: {accuracy}")
    

    total_correct=0
    total=0
    classes = load_class_names('Image classifiers\ResNet\cifar100_classes.txt')
    for data,label,orglabel in dataloader:
        pr1=max(results[0]["probabilities"][total])
        pr2=max(results[1]["probabilities"][total])
        pr3=max(results[2]["probabilities"][total])
        pr_row=[pr1,pr2,pr3]
        elected=np.argmax(pr_row)
        
        true_labels=results[elected]["true_labels"][total]
        predictedlabel=orginallabels[elected][results[elected]["predicted_labels"][total]]
        total+=1
        
        pic(classes,true_labels, predictedlabel, data)
        
        if (true_labels==predictedlabel):
            total_correct+=1
            
            
    print(total_correct/total)
    
    
if __name__ == "__main__":
    main()