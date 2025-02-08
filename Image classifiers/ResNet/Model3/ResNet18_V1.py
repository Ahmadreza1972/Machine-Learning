import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from CustomResNet18 import CustomResNet18
from DataLoad import DataLoad

class ModelProcess:
    def __init__(self,batch_size,learning_rate,epoch,valdata_ratio,num_classes,train_path,test_path):
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.epoch=epoch
        self.valdata_ratio=valdata_ratio
        self.num_classes=num_classes
        self.train_path=train_path
        self.test_path=test_path
        

    def train_model(self,model, train_loader,val_loader, criterion, optimizer, device='cpu'):
        model.to(device)
        tr_los=[]
        tr_ac=[]
        val_los=[]
        val_ac=[]
        for epoch in range(self.epoch):
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epoch}"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            tr_los.append(running_loss/len(train_loader))
            tr_ac.append(accuracy)


            model.eval()  # Set model to evaluation mode
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0

            with torch.no_grad():  # No gradients needed for validation
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            val_los.append(val_running_loss/len(val_loader))
            val_ac.append(val_accuracy)

            print(f"Tr_Loss: {running_loss/len(train_loader):.4f}, val_loss: {val_running_loss/len(val_loader):.4f}, Tr_acc: {accuracy}, val_ac: {val_accuracy}")
        return tr_ac,val_ac,tr_los,val_los

    def test_model(self,model, test_loader, criterion, device='cpu'):
        model.to(device)
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():  # No gradients needed for testing
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Loss: {running_loss/len(test_loader):.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")



# Main script
def main():
    
    train_path = r"Image classifiers\ResNet\Model3\model3_test.pth"  
    test_path = r"Image classifiers\ResNet\Model3\model3_test.pth"
    save_path= r"Image classifiers\ResNet\Model3\model_weights3.pth"
    batch_size = 16
    learning_rate = 0.001
    epoch = 35
    valdata_ratio=0.1
    num_classes=5
    
    # Check for CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-------Using device: {device}")

    Loader=DataLoad(train_path,test_path,valdata_ratio,batch_size)
    train_loader,val_loader,test_loader=Loader.DataLoad()
    
    model = CustomResNet18(num_classes=num_classes, freeze_layers=False)
    print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-------Number of trainable parameters: {trainable_params}")
    
    run=ModelProcess(batch_size, learning_rate ,epoch,valdata_ratio,num_classes,train_path,test_path)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    tr_ac,val_ac,tr_los,val_los=run.train_model(model, train_loader,val_loader, criterion, optimizer, device=device)
    torch.save(model.state_dict(), save_path)
    fig,ax=plt.subplots(ncols=2)
    ax[0].plot(tr_ac, label="Train_ac")
    ax[0].plot(val_ac, label="val_ac")
    ax[0].legend()
    ax[1].plot(tr_los, label="Train_los")
    ax[1].plot(val_los, label="val_los")
    ax[1].legend()
    plt.show()

    # Test the model
    run.test_model(model, test_loader, criterion, device=device)

if __name__ == "__main__":
    main()
