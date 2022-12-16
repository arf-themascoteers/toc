import torch
from toc_dataset import TocDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score


def test(device):
    batch_size = 10
    cid = TocDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/machine.h5")
    model.eval()
    model.to(device)
    ys = []
    yhats = []
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        simple_y = y.detach().cpu().numpy()
        simple_yhat = y_hat.detach().cpu().numpy()
        r2 = r2_score(simple_y, simple_yhat)
        print("This r2", r2)
        for ay in simple_y:
            ys.append(ay)
        for ayhat in simple_yhat:
            yhats.append(ayhat)

    r2 = r2_score(ys, yhats)
    print("Final r2",r2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
