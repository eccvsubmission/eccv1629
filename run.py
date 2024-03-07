from utils.dataloader import dataset
from utils.models import MM_MultiModalVMAE_shared_encoder,MultiModalVMAE,OneStreamVMAE,OneStreamVMAE_UnlabeledHybrid_k400ft,OneStreamSWIN,OneStreamMVIT,OneStreamVMAE_UnlabeledHybrid,  freeze_params
from torch.utils.data import DataLoader
from utils.main import run_training
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for your model.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--datasets', nargs='+', default=[], help='List of datasets to use')
    parser.add_argument('--test_datasets', nargs='+', default=[], help='List of datasets to use')
    parser.add_argument('--modalities', nargs='+', default=[], help='List of modalities to use')
    parser.add_argument('--model_type', type=str, default='OneStreamVMAE', choices=["OneStreamVMAE","MM_MultiModalVMAE_shared_encoder",
                                                                                     "OneStreamMvit", "OneStreamSwin", "MultiModalVMAE"], help='Type of model to use')
    parser.add_argument('--fusion_type', type=str, default='af', choices=["af","add","lrtf",
                                                                          "mean","deepset", "concat_w_linear"], help='Type of fusion layer to use')
    parser.add_argument('--b4c_fold',  type=int, default=0, help='Fold num for B4C dataset')
    parser.add_argument('--oxford_fold',  type=int, default=0, help='Fold num for Oxford dataset')
    parser.add_argument('--action_rec',  type=int, default=0, help='0==false,1==true')
    parser.add_argument('--minvid_length',  type=int, default=16, help='Minimum number of frames per video')
    parser.add_argument('--multilabels',  type=int, default=0, help='0==false,1==true')
    parser.add_argument('--overlapping_labels',  type=int, default=1, help='0==false,1==true, use of overlapping labels between datasets')    
    parser.add_argument('--batch_size',  type=int, default=4, help='If multi-modal video modal on T4 GPU >> set to 2, else 4 should work')
  
    args = parser.parse_args()
    # lazy boolean logic implementation
    if args.action_rec == 0: args.action_rec = False 
    else: args.action_rec = True
    if args.multilabels == 0: args.multilabels = False 
    else: args.multilabels = True
    if args.overlapping_labels == 0: args.multilabels = False 
    else: args.overlapping_labels = True
    if len(args.modalities) >1: args.multi_modal =True
    else: args.multi_modal = False
    return args #parser.parse_args()

if __name__ == '__main__': 
    args = parse_args()
    
    train_dataset = dataset(datasets=args.datasets,
                               split_type="train", 
                               b4c_fold=args.b4c_fold,
                               oxford_fold=args.oxford_fold,
                               action_rec=args.action_rec,
                               multi_modal=args.multi_modal,
                               multi_labels=args.multilabels,
                               overlapping_labels=args.overlapping_labels,
                               minvid_length=args.minvid_length)

                            
    num_classes = train_dataset.num_classes
    id2label = train_dataset.id2label
    label2id = train_dataset.label2id

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,)

    val_loaders_dict = {}
    for i, val_dataset_name in enumerate(args.test_datasets):
        val_dataset = dataset(datasets=val_dataset_name,
                               split_type="test", 
                               b4c_fold=args.b4c_fold,
                               oxford_fold=args.oxford_fold,
                               action_rec=args.action_rec,
                               multi_modal=args.multi_modal,
                               multi_labels=args.multilabels,
                               overlapping_labels=args.overlapping_labels,
                               minvid_length=args.minvid_length,
                               label2id=label2id,
                               id2label=id2label,
                               num_classes=num_classes,)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,)
        val_loaders_dict[val_dataset_name] = val_loader

    if args.model_type == "OneStreamVMAE":
        model = OneStreamVMAE(label2id=label2id, id2label=id2label)
    elif args.model_type == "OneStreamMvit":
        model = OneStreamMVIT(num_classes=num_classes)
    elif args.model_type == "OneStreamSwin":
        model = OneStreamSWIN(num_classes=num_classes)
    elif (args.model_type == "MM_MultiModalVMAE_shared_encoder") and (args.multi_modal == True):
        model = MM_MultiModalVMAE_shared_encoder(num_classes=num_classes)
        
    freeze_params(model)
    run_training(args.batch_size, train_loader, val_loaders_dict,  
                 model, num_epochs=args.num_epochs, lr=args.lr, fusion_type=args.fusion_type)
