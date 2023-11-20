import torch
from torch import nn
import sys
from src import models
from src.utils import *
import time
import os
import pickle
from tqdm import tqdm
from src.eval_metrics import *
from src.models import MM_fusion_model, MM_coordination_model


def initiate(hyp_params, train_loader, valid_loader, test_loader):

    if hyp_params.model_name == "fusion":
        model = MM_fusion_model(hyp_params)
    elif hyp_params.model_name == "coordination":
        model = MM_coordination_model(hyp_params)
    else :
        raise NotImplementedError('Incorrect model name: {}!'.format(hyp_params.model_name))
    if hyp_params.use_cuda:
        model = model.cuda()

    criterion = getattr(nn, hyp_params.criterion)()  # L1Loss used to regress

    # The optimizer and scheduler can be changed to suit your model
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp_params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when_decay, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']


    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(tqdm(train_loader)):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1

            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
            batch_size = text.size(0)

            net = nn.DataParallel(model) if batch_size > 50 else model

            preds = net(text, audio, vision)
            loss = criterion(preds, eval_attr)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            epoch_loss += loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
                batch_size = text.size(0)

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

                preds = model(text, audio, vision)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)

        if val_loss < best_valid:
            print(f"Saved model at results/{hyp_params.model_name}.pt!")
            save_model(hyp_params, model, name=hyp_params.model_name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.model_name)
    _, results, truths = evaluate(model, criterion, test=True)


    eval_mosi(results, truths, True)
