import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from math import ceil
from accelerate import Accelerator
import os
from glob import glob

from .train import train_loop
from .eval import eval_loop
from .tools import make_results_folder, store_results, save_parameters_to_txt

def run_training(
        batch_size,
        train_loader,
        val_loader,
        model,
        desired_bs=8,
        num_epochs=200,
        lr=5e-5,
        wd=0.05,
        warmup_steps=100,
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.0),
        early_stopping=25,
        verbose:bool=False,):
    # create resuls folder
    folder_path = make_results_folder()
    param_path = f"{folder_path}/parameters.txt"
    save_parameters_to_txt(param_path,
                           desired_bs=desired_bs,
                           num_epochs=num_epochs,
                           lr=lr,wd=wd,
                           warmup_steps=warmup_steps,
                           early_stopping=early_stopping,
                           model_name=model.__class__.__name__,
                           included_datasets=train_loader.dataset.included_datasets,
                           test_datasets=list(val_loader.keys()),
                           b4c_fold=train_loader.dataset.b4c_fold,
                           oxford_fold=train_loader.dataset.oxford_fold,
                           id2label=train_loader.dataset.id2label,
                           action_recognition=train_loader.dataset.action_rec)
    # setup torch params
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=wd)
    gas = ceil(desired_bs//batch_size)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_loader) * num_epochs // gas),
    )
    
    accelerator = Accelerator(gradient_accumulation_steps=gas)
    device = accelerator.device
    model = model.to(device)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler)
        
    for epoch in range(1,num_epochs+1):
        model, optimizer, scheduler, result_dict  = train_loop(model, train_loader, criterion, optimizer, 
                                                            accelerator, scheduler=lr_scheduler, verbose=verbose)    
        
        model = model.to(device)
        val_loss = 0
        if isinstance(val_loader, dict):
            for k,v in val_loader.items():
                if (k == "hdd") and (k not in train_loader.dataset.included_datasets):
                    if( epoch % 10 == 0):
                        val_result_dict =  eval_loop(model, v, criterion, verbose=verbose) 
                        store_results(f"{folder_path}{k}_", epoch, val_result_dict)
                    else:
                        pass
                else:
                    val_result_dict =  eval_loop(model, v, criterion, verbose=verbose) 
                    store_results(f"{folder_path}{k}_", epoch, val_result_dict)

                ## basically only select the 'best' model based on the training data evaluation
                if len(train_loader.dataset.included_datasets) == 1:
                    if k in train_loader.dataset.included_datasets:
                        val_loss = val_result_dict["loss"]
                else:
                    val_loss += val_result_dict["loss"]
                
            if len(train_loader.dataset.included_datasets) != 1:
                val_loss = val_loss/len(val_loader)
        else:
            val_result_dict =  eval_loop(model, val_loader, criterion, verbose=verbose)        
            store_results(folder_path, epoch, val_result_dict)
            val_loss = val_result_dict["loss"]

        if (epoch == 1) or (val_loss < best_loss):
            best_loss =  val_loss
            print(f"New best val loss: Epoch #{str(epoch)}: {round(best_loss, 4)}")
            early_stopping_count = 0
            model_path =  f"{folder_path}/best_model.pth"
            torch.save(model, model_path)
        else:
            early_stopping_count += 1
            if early_stopping_count == early_stopping:
                break
