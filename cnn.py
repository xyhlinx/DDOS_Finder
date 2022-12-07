import pandas
import torch
import seaborn
import torchmetrics
from torchsummary import summary
from matplotlib import pyplot

model_save_path = './cnn_model.pth'
dataset_filename = './processed_dataset.csv'
# env: local or colab
env = 'local'
# mode: train or validation
mode = 'validation'
validated_model = './cnn_model.pth_0.001_1129_reg95'
debug = False


def set_env(env=env, debug=debug):
    if debug:
        pd.set_option('display.max_columns', None)
    global loader
    loader = globals()['loader_' + env]
    if not validated_model and mode == 'validation':
        raise Exception('validated_model not defined. Must give a validated_model')


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=16, out_channels=32,
            kernel_size=(8,), stride=(1,), padding=(0,), dilation=(1,))
        self.leakyrelu1 = torch.nn.LeakyReLU()
        self.pool1 = torch.nn.MaxPool1d(kernel_size=5, padding=2, stride=5)

        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=256,
            kernel_size=(4,), stride=(1,), padding=(0,), dilation=(1,))
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2, padding=0, stride=2)

        self.conv3 = torch.nn.Conv1d(in_channels=256, out_channels=512,
            kernel_size=(3,), stride=(1,), padding=(0,), dilation=(1,))
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool1d(kernel_size=2, padding=0, stride=2)

        self.fc1 = torch.nn.Linear(in_features=512, out_features=64)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.leakyrelu2 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.dropout2 = torch.nn.Dropout(p=0.1)
        self.leakyrelu3 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(in_features=64, out_features=16)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        # convolution
        # print(x.shape)
        o = self.conv1(x)
        o = self.leakyrelu1(o)
        # print(o.size())
        # print(o)

        # pooling
        o = self.pool1(o)
        # print(o.size())

        # convolution
        o = self.conv2(o)
        # print(o.size())
        o = self.relu2(o)

        # pooling
        o = self.pool2(o)

        # convolution
        o = self.conv3(o)
        # print(o.size())
        o = self.relu3(o)

        # pooling
        o = self.pool3(o)
        # print(o.size())

        # linear layer
        o = torch.flatten(o, start_dim=0, end_dim=-1)
        # print(o.size())
        o = self.fc1(o)
        o = self.dropout1(o)
        o = self.leakyrelu2(o)

        # o = o.view(o.size(0), -1)
        # print(o.size())
        o = self.fc2(o)
        o = self.dropout2(o)
        o = self.leakyrelu3(o)

        # print(o)
        o = self.fc3(o)
        # print(o.size())
        o = self.sigmoid1(o)
        # print(o)
        # print(o.size())

        return o


def evaluate(predictions, labels):
    predictions = (predictions >= (1 / len(predictions))).float()
    true_pos_neg = torch.eq(predictions, labels).float()
    acc = true_pos_neg.sum() / predictions.numel()
    # print(true_pos_neg.sum(), predictions.numel())
    # print(predictions)
    # print(labels)
    return acc


def manipulate_data(df):
    # oh_src = pandas.get_dummies(df['src_ip'], prefix='src')
    # oh_dst = pandas.get_dummies(df['dst_ip'], prefix='dst')

    # trim the data to the equal amount
    positive_quantity = (df['Label'] == 1).sum()
    negative_quantity = (df['Label'] == 0).sum()
    if negative_quantity > positive_quantity:
        indices = df.loc[df['Label'] == 0].sample(negative_quantity - positive_quantity).index
    else:
        indices = df.loc[df['Label'] == 1].sample(positive_quantity - negative_quantity).index
    df = df.drop(indices)
    print((df['Label'] == 1).sum(), (df['Label'] == 0).sum(), positive_quantity, negative_quantity)

    y = df['Label']
    df = df.drop(columns='Label')
    df = (df - df.mean(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    df = df.dropna(axis=1)
    df['Label'] = y
    ts = torch.tensor(df.values)
    # ts = torch.unsqueeze(ts, dim=2)
    # ts = torch.unsqueeze(ts, dim=1)
    # ts = torch.squeeze(ts, dim=3)
    # print(df.isnull())
    return ts


def loader_local(filename):
    # df = pandas.read_csv(filename, names=['time', 'length', 'src_ip', 'dst_ip', 'response_time', 'y'])
    df = pandas.read_csv(filename, header=0)
    return df


def loader_colab(filename):
    try:
        from google.colab import files
        uploaded = files.upload()
    except:
        return


def step_iter(start, stride, numbers=None, end=None):
    if not end and not numbers:
        return
    if end:
        numbers = int((end - start) // stride)

    for i in range(0, numbers):
        yield start + i * stride


def train(train_loader, validation_loader, lr, epochs=3, model_save_path='./model.pth'):
    net = ConvNet()
    print(net)
    net.float()
    # summary(net, (1, 3))
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print('lr: {:f}'.format(lr))
    for i in range(epochs):

        # train
        cumulated_loss = 0.0
        training_counter = 0
        for j, data in enumerate(train_loader):
            if len(data) < train_loader.batch_size:
                break
            training_data = data[:, :-1]
            y = data[:, -1]
            y = torch.reshape(y, (-1,))
            y_prediction = net(training_data.float())
            y_prediction = torch.reshape(y_prediction, (-1,))
            # print(y_prediction)
            loss = criterion(y_prediction, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumulated_loss += loss.item()
            training_counter += 1

        # validation
        report = validation(validation_loader, net)

        # report
        print('epoch: {}, {}/{}, loss: {}, validation_loss: {}, accuracy: {}'
              .format((i + 1),
                      (i + 1) * training_counter,
                      epochs * training_counter,
                      cumulated_loss / training_counter,
                      report['loss'],
                      report['accuracy']))

    # pyplot.plot([i for i in range(1, len(report) + 1)], report, label=str(lr))
    # pyplot.savefig('{:.3f}_lr.png'.format(lr))
    # pyplot.show()

    torch.save(net.state_dict(), model_save_path + '_' + str(lr))


def validation(validation_loader, model):
    report = {}

    test_cumulated_loss = 0.0
    validation_counter = 0
    correct = 0
    criterion = torch.nn.BCELoss(reduction='mean')
    confmat = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)
    confres = torch.tensor([[0, 0], [0, 0]])

    for k, data in enumerate(validation_loader):
        if len(data) < validation_loader.batch_size:
            break
        data, y = data[:, 0:-1], data[:, -1]
        y = torch.reshape(y, (-1,))
        pred = model(data.float())
        pred = torch.reshape(pred, (-1,))
        discrete_pred = (pred >= 0.5)
        correct += (discrete_pred == y).sum() / validation_loader.batch_size
        test_cumulated_loss += criterion(pred, y.float())
        validation_counter += 1
        confres += confmat(discrete_pred, y)

    accuracy = correct.sum() / validation_counter
    loss = test_cumulated_loss / validation_counter
    report['accuracy'] = accuracy.item()
    report['loss'] = loss.item()
    report['confusion'] = confres
    return report


if __name__ == '__main__':
    set_env(env, debug)
    batch_size = 16
    df = loader(dataset_filename)
    ts = manipulate_data(df)

    train_size = int(0.1 * len(ts))
    test_size = int(len(ts) - train_size)
    train_data, val_data = torch.utils.data.random_split(ts, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=6)
    validation_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=6)

    print('train_data rows: {}, batch size: {}, mini-batches: {}'.format(
        len(train_loader.dataset),
        batch_size,
        len(train_loader.dataset) // batch_size))
    print('validation_data rows: {}, batch size: {}, mini-batches: {}'.format(
        len(validation_loader.dataset),
        batch_size,
        len(validation_loader.dataset) // batch_size))

    step = step_iter(start=0.001, numbers=1, stride=1e-3)
    while mode == 'train':
        try:
            train(train_loader, validation_loader, lr=next(step), epochs=200,
                  model_save_path=model_save_path)
        except StopIteration:
            break

    if mode == 'validation':
        net = ConvNet()
        net.load_state_dict(torch.load(validated_model))
        report = validation(validation_loader, net)
        print('validation size: {}, loss: {}, accuracy: {}, confusion: {}'.format(
            len(validation_loader.dataset),
            report['loss'],
            report['accuracy'],
            report['confusion']))

        seaborn.heatmap(report['confusion'], annot=True)
    pyplot.show()
    
