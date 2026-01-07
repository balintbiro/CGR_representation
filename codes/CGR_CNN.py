import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("../data/ACDEFGHIKLMNPQRSTVWY_0865_35.csv")
X,y=df.drop(columns=["label"]).values.astype("float32"),df["label"].values.astype("float32")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
XCnn = X.reshape(-1, 1, 35,35)

XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)
    
class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(10 * 16 * 16, 1)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    
torch.manual_seed(0)

cnn = NeuralNetBinaryClassifier(
    Cnn,
    max_epochs=10,
    lr=0.001,
    optimizer=torch.optim.Adam,
    device=device,
)

cnn.fit(XCnn_train, y_train)

last_acc=cnn.history[-1]["valid_acc"]
