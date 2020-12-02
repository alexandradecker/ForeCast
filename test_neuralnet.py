import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from datetime import datetime
from dataloader import get_data_loader, MyDataset


def test(args, device):
    full_data = get_data_loader(args)

    if args.model_type == "CNN":
        from CNN import CNN
        model = CNN(args).to(device)
    elif args.model_type == "MLP":
        from MLP import MLP
        model = MLP(args).to(device)
    elif args.model_type == "LSTM":
        from LSTM import LSTM
        model = LSTM(args).to(device)

    optimiser = optim.Adam(
        model.parameters(), lr=args.learning_rate)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state['model'])
    optimiser.load_state_dict(state['optimiser'])

    total_difference = 0
    n = 0

    for batch_num, data in enumerate(full_data):
        x, y = data[0].float().to(device), data[1].float().to(device)
        num_of_predictions = x.shape[0]
        pred = model(x)
        pred = pred.reshape(y.shape)
        for i in range(num_of_predictions):
            if pred[i] >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
        total_difference += sum(abs(pred - y))
        n += num_of_predictions

        del x
        del y

    return 1 - (total_difference/n).item()


if __name__ == '__main__':
    from attrdict import AttrDict
    args = AttrDict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args_dict = {
        'learning_rate': 0.0001,
        'batch_size': 8,
        'num_epochs': 10000,
        'input_dim': 11,
        'output_dim': 1,
        'num_layers': 3,
        'num_conv_layers': 6,  # num of convolution layers will be 1 less than this
        'conv_channels': [1, 2, 4, 8, 16, 1],  # length same as number above
        'perceptrons_per_layer': 10,
        'perceptrons_in_conv_layers': 10,
        'res_channel': 64,
        'num_residual_layers': 0,
        'load_models': False,
        'model_path': "models/CNN/model2020-12-02 12:04:12.348098.pt",
        'activation': nn.ReLU,
        'norm': nn.BatchNorm1d,
        'loss_function': nn.MSELoss,
        'save_path': "models/CNN",
        'use_wandb': False,
        'dropout': False,
        'decay': True,
        'test': False,
        'model_type': "CNN",
        'participant': None,
        'device': device
    }
    args.update(args_dict)

    print("Accuracy is {}".format(test(args, device)))
    with open(args.model_path[:-3] + ".txt", 'w') as file:
        file.write(str(args_dict))
