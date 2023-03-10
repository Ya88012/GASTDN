import sys

import keras
import numpy as np
# import xgboost as xgb
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.backend import set_session

import STDN_master.file_loader as file_loader
import STDN_master.models as models
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping
import datetime
import argparse
import gc

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--dataset', type=str, default='bike', help='taxi or bike')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch')
parser.add_argument('--max_epochs', type=int, default=1000,
                    help='maximum epochs')
parser.add_argument('--att_lstm_num', type=int, default=3,
                    help='the number of time for attention (i.e., value of Q in the paper)')
parser.add_argument('--long_term_lstm_seq_len', type=int, default=3,
                    help='the number of days for attention mechanism (i.e., value of P in the paper)')
parser.add_argument('--short_term_lstm_seq_len', type=int, default=7,
                    help='the length of short term value')
parser.add_argument('--cnn_nbhd_size', type=int, default=3,
                    help='neighbors for local cnn (2*cnn_nbhd_size+1) for area size')
parser.add_argument('--nbhd_size', type=int, default=2,
                    help='for feature extraction')
parser.add_argument('--cnn_flat_size', type=int, default=128,
                    help='dimension of local conv output')
parser.add_argument('--model_name', type=str, default='stdn',
                    help='model name')

args = parser.parse_args()
# print(args)

class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1
    mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))

    return rmse, mape


def eval_lstm(y, pred_y, threshold):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(
            np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)


def main(batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):
    model_hdf5_path = "./hdf5s/"

    if args.dataset == 'taxi':
        sampler = file_loader.file_loader()
    elif args.dataset == 'bike':
        sampler = file_loader.file_loader(config_path = "data_bike.json")
    else:
        raise Exception("Can not recognize dataset, please enter taxi or bike")
    modeler = models.models()

    if args.model_name == "stdn":

        tf.keras.backend.clear_session()
        # training
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype="train",
                                                                          att_lstm_num=args.att_lstm_num, \
                                                                          long_term_lstm_seq_len=args.long_term_lstm_seq_len,
                                                                          short_term_lstm_seq_len=args.short_term_lstm_seq_len, \
                                                                          nbhd_size=args.nbhd_size,
                                                                          cnn_nbhd_size=args.cnn_nbhd_size)

        print("Start training {0} with input shape {2} / {1}".format(args.model_name, x.shape, cnnx[0].shape))

        model = modeler.stdn(att_lstm_num=args.att_lstm_num, att_lstm_seq_len=args.long_term_lstm_seq_len, \
                             lstm_seq_len=len(cnnx), feature_vec_len=x.shape[-1], \
                             cnn_flat_size=args.cnn_flat_size, nbhd_size=cnnx[0].shape[1], nbhd_type=cnnx[0].shape[-1])

        model.fit( \
            x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ], \
            y=y, \
            batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop])

        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype="test", nbhd_size=args.nbhd_size,
                                                                          cnn_nbhd_size=args.cnn_nbhd_size)
        y_pred = model.predict( \
            x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ], )
        threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
        print("Evaluating threshold: {0}.".format(threshold))
        (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
        print(
            "Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(
                args.model_name, prmse, pmape * 100, drmse, dmape * 100))

        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model.save(model_hdf5_path + args.model_name + currTime + ".hdf5")
        return

    else:
        print("Cannot recognize parameter...")
        return


def get_fitness(indi, batch_size = 64, max_epochs = 3, validation_split = 0.2, early_stop = EarlyStopping()):
    model_hdf5_path = "./hdf5s/"

    print("In STDN.py's get fitness!!!!!")
    print('Now trying to get indi.arg_list fitness:', indi.arg_list)
    print("att_lstm_num:", indi.arg_list[0])
    print("long_term_lstm_seq_len:", indi.arg_list[1])
    print("short_term_lstm_seq_len:", indi.arg_list[2])
    print("nbhd_size:", indi.arg_list[3])
    print("cnn_nbhd_size:", indi.arg_list[4])

    if args.dataset == 'taxi':
        sampler = file_loader.file_loader()
    elif args.dataset == 'bike':
        sampler = file_loader.file_loader(config_path = "STDN_master/data_bike.json")
    else:
        raise Exception("Can not recognize dataset, please enter taxi or bike")
    modeler = models.models()

    if args.model_name == "stdn":
        # training
        tf.keras.backend.clear_session()
        print("Now try to training sample.")
        
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype = "train",
                                                                          att_lstm_num = indi.arg_list[0], \
                                                                          long_term_lstm_seq_len = indi.arg_list[1],
                                                                          short_term_lstm_seq_len = indi.arg_list[2], \
                                                                          nbhd_size = indi.arg_list[3],
                                                                          cnn_nbhd_size = indi.arg_list[4])

        print("Start training {0} with input shape {2} / {1}".format(args.model_name, x.shape, cnnx[0].shape))

        model = modeler.stdn(att_lstm_num = indi.arg_list[0], att_lstm_seq_len = indi.arg_list[1], \
                             lstm_seq_len = len(cnnx), feature_vec_len = x.shape[-1], \
                             cnn_flat_size = args.cnn_flat_size, nbhd_size = cnnx[0].shape[1], nbhd_type = cnnx[0].shape[-1])

        model.fit( \
            x = att_cnnx + att_flow + att_x + cnnx + flow + [x, ], \
            y = y, \
            batch_size = batch_size, validation_split = validation_split, epochs = max_epochs, callbacks = [early_stop])

        del att_cnnx, att_flow, att_x, cnnx, flow, x, y
        gc.collect()

        print("Finish the training and validation.")

        print("Now try to testing sample.")
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype = "test", 
                                                                          att_lstm_num = indi.arg_list[0], \
                                                                          long_term_lstm_seq_len = indi.arg_list[1],
                                                                          short_term_lstm_seq_len = indi.arg_list[2], \
                                                                          nbhd_size = indi.arg_list[3],
                                                                          cnn_nbhd_size = indi.arg_list[4])
        y_pred = model.predict( \
            x = att_cnnx + att_flow + att_x + cnnx + flow + [x, ], )
        threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
        print("Finish testing/predicting.")
        print("Evaluating threshold: {0}.".format(threshold))
        (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
        print(
            "Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(
                args.model_name, prmse, pmape * 100, drmse, dmape * 100))

        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        argument_list = '_' + str(indi.arg_list[0]) + '_' + str(indi.arg_list[1]) + '_' + str(indi.arg_list[2]) + '_' + str(indi.arg_list[3]) + '_' + str(indi.arg_list[4])
        model.save(model_hdf5_path + args.model_name + currTime + argument_list + ".hdf5")

        del att_cnnx, att_flow, att_x, cnnx, flow, x, y
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return (prmse * 100, pmape * 100), (drmse * 100, dmape * 100)

    else:
        print("Cannot recognize parameter...")
        return ((-1.0, -1.0), (-1.0, -1.0))

def get_final_output(indi, batch_size = 64, max_epochs = 1000, validation_split = 0.2, early_stop = EarlyStopping()):
    model_hdf5_path = "./hdf5s/"

    print('Now trying to get indi.arg_list fitness:', indi.arg_list)
    print("att_lstm_num:", indi.arg_list[0])
    print("long_term_lstm_seq_len:", indi.arg_list[1])
    print("short_term_lstm_seq_len:", indi.arg_list[2])
    print("nbhd_size:", indi.arg_list[3])
    print("cnn_nbhd_size:", indi.arg_list[4])

    if args.dataset == 'taxi':
        sampler = file_loader.file_loader()
    elif args.dataset == 'bike':
        sampler = file_loader.file_loader(config_path = "STDN_master/data_bike.json")
    else:
        raise Exception("Can not recognize dataset, please enter taxi or bike")
    modeler = models.models()

    if args.model_name == "stdn":
        tf.keras.backend.clear_session()
        # training
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype = "train",
                                                                          att_lstm_num = indi.arg_list[0], \
                                                                          long_term_lstm_seq_len = indi.arg_list[1],
                                                                          short_term_lstm_seq_len = indi.arg_list[2], \
                                                                          nbhd_size = indi.arg_list[3],
                                                                          cnn_nbhd_size = indi.arg_list[4])

        print("Start training {0} with input shape {2} / {1}".format(args.model_name, x.shape, cnnx[0].shape))

        model = modeler.stdn(att_lstm_num = indi.arg_list[0], att_lstm_seq_len = indi.arg_list[1], \
                             lstm_seq_len = len(cnnx), feature_vec_len = x.shape[-1], \
                             cnn_flat_size = args.cnn_flat_size, nbhd_size = cnnx[0].shape[1], nbhd_type = cnnx[0].shape[-1])

        model.fit( \
            x = att_cnnx + att_flow + att_x + cnnx + flow + [x, ], \
            y = y, \
            batch_size = batch_size, validation_split = validation_split, epochs = max_epochs, callbacks = [early_stop])

        del att_cnnx, att_flow, att_x, cnnx, flow, x, y
        gc.collect()

        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype = "test", 
                                                                          att_lstm_num = indi.arg_list[0], \
                                                                          long_term_lstm_seq_len = indi.arg_list[1],
                                                                          short_term_lstm_seq_len = indi.arg_list[2], \
                                                                          nbhd_size = indi.arg_list[3],
                                                                          cnn_nbhd_size = indi.arg_list[4])
        y_pred = model.predict( \
            x = att_cnnx + att_flow + att_x + cnnx + flow + [x, ], )
        threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
        print("Evaluating threshold: {0}.".format(threshold))
        (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
        print(
            "Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(
                args.model_name, prmse, pmape * 100, drmse, dmape * 100))

        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        argument_list = '_' + str(indi.arg_list[0]) + '_' + str(indi.arg_list[1]) + '_' + str(indi.arg_list[2]) + '_' + str(indi.arg_list[3]) + '_' + str(indi.arg_list[4])
        model.save(model_hdf5_path + args.model_name + currTime + argument_list + "_final" + ".hdf5")

        del att_cnnx, att_flow, att_x, cnnx, flow, x, y
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return (prmse * 100, pmape * 100), (drmse * 100, dmape * 100)

    else:
        print("Cannot recognize parameter...")
        return ((-1.0, -1.0), (-1.0, -1.0))

if __name__ == "__main__":
    stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)
    main(batch_size=args.batch_size, max_epochs=args.max_epochs, early_stop=stop)
