from torch.utils.data import Dataset
import pickle
import numpy as np
import torch
import torch.nn as nn


def load_pickle(file_dir):
    """Load from pickle file.

    Args:
        file_dir (BinaryIO): dir of the pickle file.

    Returns:
        any type: the loaded data.
    """
    with open(file_dir, 'rb') as file:
        try:
            data = pickle.load(file, encoding='latin-1')
        except Exception:
            data = pickle.load(file)

    return data


def build_input_from_pkl(device, source_dir: str, split: str):
    """
        Args:
            split (str, optional): denote the train, dev and test set.
    """
    data = load_pickle(source_dir)
    num_event_types = data["dim_process"]
    source_data = data[split]
    time_seqs = [[float(x["time_since_start"]) for x in seq] for seq in source_data if seq]
    type_seqs = [[x["type_event"] for x in seq] for seq in source_data if seq]
    type_seqs = [torch.tensor(type_seqs[i], device=device) for i in range(len(type_seqs))]

    mins = [min(seq) for seq in time_seqs]
    time_seqs = [[round(time - min_val, 6) for time in time_seq] for time_seq, min_val in zip(time_seqs, mins)]
    time_seqs = [torch.tensor(time_seqs[i], device=device) for i in range(len(time_seqs))]

    seqs_lengths = [torch.tensor(len(seq), device=device) for seq in time_seqs]

    return time_seqs, type_seqs, num_event_types, seqs_lengths


def process_loaded_sequences(device, source_dir: str, split: str):
    """
    Preprocess the dataset by padding the sequences.
    """

    time_seqs, type_seqs, num_event_types, seqs_lengths = \
        build_input_from_pkl(device, source_dir, split)

    tmax = max([max(seq) for seq in time_seqs])

    #  Build a data tensor by padding
    time_seqs = nn.utils.rnn.pad_sequence(time_seqs, batch_first=True, padding_value=tmax + 1)
    type_seqs = nn.utils.rnn.pad_sequence(type_seqs, batch_first=True, padding_value=0)
    mask = (time_seqs != tmax + 1).float()

    return time_seqs, type_seqs, num_event_types, seqs_lengths, mask


class SeqDataset(Dataset):

    def __init__(self, train_time, train_type, train_num_types, train_seq_lengths, train_mask):
        super().__init__()

        self.train_time = train_time
        self.train_type = train_type
        self.train_num_types = train_num_types
        self.train_seq_lengths = train_seq_lengths
        self.train_mask = train_mask

    def __len__(self):
        return len(self.train_time)

    def __getitem__(self, index):

        return self.train_time[index], self.train_type[index], self.train_mask[index]

def load_dataset(dataname, device):
    '''
    For now we consider 4 type dataset in our experiments
    Taobao, NYMVC, UBUNTU, MATHOF
    '''

    if dataname in ['nymvc', 'taobao', 'mathof', 'ubuntu', 'taxi']:
        train_time, train_type, train_num_types, train_seq_lengths, train_mask = \
            process_loaded_sequences(device=device, source_dir='./data/{}/train.pkl'.format(dataname), split='train')

        dev_time, dev_type, dev_num_types, dev_seq_lengths, dev_mask = \
            process_loaded_sequences(device=device, source_dir='./data/{}/dev.pkl'.format(dataname), split='dev')

        train_dataset = SeqDataset(train_time, train_type, train_num_types, train_seq_lengths, train_mask)
        dev_dataset = SeqDataset(dev_time, dev_type, dev_num_types, dev_seq_lengths, dev_mask)

        test_time, test_type, test_num_types, test_seqs_lengths = \
            build_input_from_pkl(device=device, source_dir='./data/{}/test.pkl'.format(dataname),
                                 split='test')
        num_test_event = sum([len(seq) for seq in test_time])

    else:
        raise Exception("Should choose a right dataset name !!!!!!")

    return train_dataset, dev_dataset, train_num_types, test_time, test_type, num_test_event

