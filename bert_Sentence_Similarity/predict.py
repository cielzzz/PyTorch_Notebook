# -*- coding: utf-8 -*-

import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from BertModel import MyBertModelTest
from utils import test
from data_process import DataPrecessForSentence

def main(test_file, pretrained_file, batch_size=32):

    device = torch.device("cuda")
    bert_tokenizer = BertTokenizer.from_pretrained('./data/models/vocabs.txt', do_lower_case=True)
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(bert_tokenizer, test_file)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing BERT model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    main("./data/test.csv", "./data/models/best.pth.tar")