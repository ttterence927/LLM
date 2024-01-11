import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from utils.metrics import metric
from models.loss import NoFussCrossEntropyLoss
from models import analysis
plt.switch_backend('agg')
import numpy as np
from scipy.stats import norm
class Config:
    def __init__(self, bin_size, max_coverage):
        self.bin_size = bin_size
        self.max_coverage = max_coverage

class Tokenizer:
    def __init__(self, config):
        self.BIN_SIZE = config.bin_size
        self.MAX_COVERAGE = config.max_coverage
        self.bins, self.bin_values = self.get_gaussian_bins(self.BIN_SIZE, self.MAX_COVERAGE)
        self.vocab_size = len(self.bin_values)


    def get_gaussian_bins(self,bin_size, max_coverage):
        # returns bin boundaries and bin centers s.t. each bin contains
        # BIN_SIZE % of the total gauusian distribution.
        N = norm(loc=0, scale=1)
        def _get_next_bin_boundary(init_pt, bin_size):
            cdf_right_boundary = np.clip(N.cdf(init_pt) + bin_size, 0, 0.99999)
            return N.ppf(cdf_right_boundary)

        pos_bins = [0]; coverage = 0
        while coverage < max_coverage / 2:
            nxt_bin_boundary = _get_next_bin_boundary(
                init_pt=pos_bins[-1],
                bin_size=bin_size)
            pos_bins.append(nxt_bin_boundary)
            coverage = N.cdf(nxt_bin_boundary) - N.cdf(0)

        all_bins = np.array([-x for x in pos_bins[1::][::-1]] + pos_bins)
        bin_center = 0.5 * (all_bins[:-1] + all_bins[1:])

        return all_bins, bin_center

    @staticmethod
    def norm_std(x, loc, scale):
        return ((x - loc) / (scale + 1e-6))

    @staticmethod
    def denorm_std(x, loc, scale):
        return (x * scale + loc)

    def clip(self, x):
        # clips all values in x  thath is smaller than the smallest 
        # bin or larger than the largest bin.
        return np.clip(x, self.bins[0]+(1e-3), self.bins[-1]-(1e-3))

    def encode(self, x, params=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        # Assuming x is a 1D time series array
        if params is None:
            params = {'loc': x.mean(), 'scale': x.std()}

        x_norm = self.norm_std(x, **params)
        x_clipped = self.clip(x_norm)
        token_ids = np.digitize(x_clipped, self.bins, right=False) - 1
        token_ids = np.clip(token_ids, 0, self.vocab_size - 1)

        return token_ids, params

    def decode(self, tkn_id, params):
        if not isinstance(tkn_id, np.ndarray):
            tkn_id = np.array(tkn_id)

        values = self.bin_values[tkn_id]
        values = self.denorm_std(values, **params)
        return values

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

def vali(model, vali_data, vali_loader, criterion,criterion_classification, args, device, itr):
    analyzer = analysis.Analyzer(print_conf_mat=True)
    total_loss = []
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.eval()
    else:
        model.in_layer.eval()
        model.out_layer.eval()
    per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)


            regression_output, classification_output = model(batch_x, i)

            regression_output = regression_output.squeeze()

            batch_y_reg = batch_y  # Regression targets
            batch_y_cls = batch_y_mark   # Classification targets, assuming last column is class label



            batch_y_reg = batch_y_reg[:, -args.pred_len:].squeeze().to(device)
            batch_y_cls = batch_y_cls.squeeze().to(device)  # Reshape and convert to long for classification criterion
            #print(regression_output, batch_y_reg)
            #print(classification_output.float(), batch_y_cls)
            loss_reg = criterion(regression_output.unsqueeze(-1), batch_y_reg.unsqueeze(-1)).cpu()
            loss_cls = criterion_classification(classification_output.squeeze().float(), batch_y_cls).cpu()

            loss = (1- args.class_ratio) * loss_reg + args.class_ratio * loss_cls  # Combine losses, you can also weigh them differently
            #print(loss_reg, loss_cls)
            total_loss.append(loss)
            per_batch['targets'].append(batch_y_cls.cpu().numpy())
            per_batch['predictions'].append(classification_output.cpu().numpy())
            per_batch['metrics'].append([loss.cpu().numpy()])

    total_loss = np.average(total_loss)
    predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
    print('pred',predictions.shape)
    probs = torch.nn.functional.softmax(predictions)  # (total_samples, num_classes) est. prob. for each class and sample
    
    predictions = torch.argmax(probs, dim=0).cpu().numpy()  # (total_samples,) int class index for each sample
    probs = probs.cpu().numpy()
    #print(probs.shape)
    print('pred',predictions)
    print(probs)
    #print(predictions)
    class_names = np.arange(probs.shape[1]) 
    targets = np.concatenate(per_batch['targets'], axis=1)#.flatten()
    targets = np.argmax(targets, axis=1)
    print(targets)

    
    if args.data == 'ttf':
        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        threshold = 0.4
        predictions = (predictions > threshold).int().cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = [0, 1]
    metrics_dict = analyzer.analyze_classification(predictions, targets, class_names)
    print("Acc: ", metrics_dict['total_accuracy'])
    print("Total Average Loss: ", total_loss)

    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.train()
    else:
        model.in_layer.train()
        model.out_layer.train()
    return total_loss
from sklearn import metrics
from tabulate import tabulate
import matplotlib.pyplot as plt
def print_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix'):
    """Print confusion matrix as text to terminal"""

    if label_strings is None:
        label_strings = ConfMat.shape[0] * ['']

    print(title)
    print(len(title) * '-')
    # Make printable matrix:
    print_mat = []
    for i, row in enumerate(ConfMat):
        print_mat.append([label_strings[i]] + list(row))
    print(tabulate(print_mat, headers=['True\Pred'] + label_strings, tablefmt='orgtbl'))
def plot_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    """Plot confusion matrix in a separate window"""
    plt.imshow(ConfMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if label_strings:
        tick_marks = np.arange(len(label_strings))
        plt.xticks(tick_marks, label_strings, rotation=90)
        plt.yticks(tick_marks, label_strings)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def test(model, test_data, test_loader, criterion, criterion_classification, args, device, itr):
    preds = []
    trues = []
    # mases = []
    per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            # outputs_np = batch_x.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_input_itr{}_{}.npy".format(itr, i), outputs_np)
            # outputs_np = batch_y.cpu().numpy()
            # np.save("emb_test/ETTh2_192_test_true_itr{}_{}.npy".format(itr, i), outputs_np)

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_y_mark = batch_y_mark.float().to(device)


            regression_output, classification_output = model(batch_x, i)

            regression_output = regression_output.squeeze()#[:, -args.pred_len:, :]

            batch_y_reg = batch_y  # Regression targets
            batch_y_cls = batch_y_mark   # Classification targets, assuming last column is class label



            batch_y_reg = batch_y_reg[:, -args.pred_len:].squeeze().to(device)
            batch_y_cls = batch_y_cls.squeeze().to(device)  # Reshape and convert to long for classification criterion
            #print(regression_output, batch_y_reg)
            #print(classification_output.float(), batch_y_cls)
            
            loss_reg = criterion(regression_output.unsqueeze(-1), batch_y_reg.unsqueeze(-1)).cpu()
            loss_cls = criterion_classification(classification_output.squeeze().float(), batch_y_cls).cpu()

            loss = (1- args.class_ratio) * loss_reg + args.class_ratio * loss_cls  # Combine losses, you can also weigh them differently
            #print(loss_reg, loss_cls)
            total_loss.append(loss)
            per_batch['targets'].append(batch_y_cls.cpu().numpy().reshape([-1,1]))
            per_batch['predictions'].append(classification_output.cpu().numpy().reshape([-1,1]))
            per_batch['metrics'].append([loss.cpu().numpy()])

            preds.append(regression_output.unsqueeze(-1))
            trues.append(batch_y_reg.unsqueeze(-1))


    total_loss = np.average(total_loss)
    predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
    print(predictions)
    probs = torch.nn.functional.softmax(predictions)  # (total_samples, num_classes) est. prob. for each class and sample
    
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    probs = probs.cpu().numpy()
    #print(probs.shape)
    print(predictions)
    print(probs)
    #print(predictions)
    class_names = np.arange(probs.shape[1]) 
    targets = np.concatenate(per_batch['targets'], axis=1)#.flatten()
    targets = np.argmax(targets, axis=1)
    print(targets)
    
    if args.data == 'ttf':
        predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
        threshold = 0.4
        predictions = (predictions > threshold).int().cpu().numpy()
        targets = np.concatenate(per_batch['targets'], axis=0).flatten()
        class_names = [0, 1]
    analyzer = analysis.Analyzer(print_conf_mat=True)
    metrics_dict = analyzer.analyze_classification(predictions, targets, class_names)
    print("Acc: ", metrics_dict['total_accuracy'])
    #print("Total Average Loss: ", total_loss)
    preds = np.array(preds)
    trues = np.array(trues)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))
    return mse, mae
