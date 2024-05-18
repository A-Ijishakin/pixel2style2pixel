import wandb 
import sys 
sys.path.append('../') 
import torch 
import numpy as np 
import torch.nn as nn  
from dataset import CelebA_Dataset 
import multiprocessing 
from torch.optim import Adam     
from metric import MultiTaskLoss 
from tqdm import tqdm 
from torch.utils.data import DataLoader 
import os 
import argparse
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  
from models.psp import pSp 
from argparse import Namespace


parser = argparse.ArgumentParser(description='CelebA Evaluation') 
parser.add_argument('--device', default='cuda:0', type=str) 
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int) 
parser.add_argument('--ext', default='', type=str) 
parser.add_argument('--check_val', default=False, type=bool) 
args = parser.parse_args()  
 
class EvalCeleba_Test():
    def __init__(self, args):
        self.args = args            
        if args.ext != 'ffhq': 
                # model_specs = {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"} 
                # self.get_download_model_command(model_specs['id'], model_specs['name']) 
                opts = torch.load('/home/rmapaij/sae_bench/pSpGAN/psp_celebs_super_resolution.pt', map_location='cpu')['opts'] 
                opts['output_size'] = 512 
                opts['checkpoint_path'] = '/home/rmapaij/sae_bench/pSpGAN/psp_celebs_super_resolution.pt'
                opts['learn_in_w'] = False
                opts = Namespace(**opts)
                model = pSp(opts) 
                self.encoder = model.encoder.to(args.device)
                
        else:
            opts = torch.load('/home/rmapaij/sae_bench/pSpGAN/psp_ffhq_encode.pt', map_location='cpu')['opts']
            opts['output_size'] = 512 
            opts['checkpoint_path'] = '/home/rmapaij/sae_bench/pSpGAN/psp_ffhq_encode.pt'
            opts['learn_in_w'] = False 
            opts['encoder_type'] = 'GradualStyleEncoder' 
            opts = Namespace(**opts)
            model = pSp(opts) 
            self.encoder = model.encoder.to(args.device)

        
        wandb.init(project="HSpace-SAEs", entity="a-ijishakin",
                        name=f'PDAE testing {args.ext}')

        self.config = wandb.config
        
    def train(self):

        
        train_loader = DataLoader(dataset = CelebA_Dataset(mode=0, dataset=self.args.ext) , batch_size=self.args.batch_size, 
                                shuffle=True, 
                                num_workers=8, 
                                persistent_workers=True) 
        
        val_loader = DataLoader(dataset = CelebA_Dataset(mode=1) , batch_size=self.args.batch_size, 
                                shuffle=True, 
                                num_workers=8, 
                                persistent_workers=True)  
        
        length = len(train_loader) 

        total_iters  = length + 60 if self.args.check_val else length 
        
        classifier = nn.Linear(512, 40).to(self.args.device)  
        optimizer = Adam(classifier.parameters(), lr=0.001)
        # define loss function
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss_fn = MultiTaskLoss(loss_fn=loss_fn)
                
        # training
        best_epoch_loss = np.inf 

        step = 0
        for epoch in range(self.args.num_epochs):
            with tqdm(total=total_iters, desc=f'Epoch {epoch}', unit='batch') as pbar:
                epoch_loss = 0 
                classifier.train()
                for index, batch in enumerate(train_loader):
                    image, labels = batch['imgs'].to(self.args.device), batch['labels'].to(self.args.device)  
                    latent = self.encoder(image) 
                    logits = classifier(latent)   
                    # loss = loss_fn(logits, labels) 
                    loss = loss_fn.compute(logits, labels, return_dict=False)
                    loss.backward()
                    optimizer.step() 
                    step += 1 
                    
                    epoch_loss += loss.item()    
                    wandb.log({"loss": loss.item()},  
                            step=(epoch * length) + index) 
                    pbar.update(1) 
                
                if self.args.check_val:
                    self.eval_multitask(val_loader, pbar=pbar,  step=(epoch*length) + index, 
                                                classifier=classifier) 
                    
                else:
                    if epoch_loss < best_epoch_loss:
                        best_epoch_loss = epoch_loss
                        self.save_classifier(classifier=classifier, type='best')     

                if epoch % 50 == 0:
                    self.save_classifier(classifier=classifier, type='all', epoch=epoch) 

                if epoch == 15:
                    self.save_classifier(classifier=classifier, type='all', epoch=epoch)  
                
                epoch_loss /= length  
                wandb.log({'Epoch Loss': 
                    epoch_loss}, 
                    step=(epoch * length) + index) 

                self.save_classifier(classifier=classifier, type='last') 
    
    
    
    def get_download_model_command(self, file_id, file_name):
        save_path = 'pretrained_models'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
        return url
        
        
    
    
    def eval_multitask(self, eval_loader, classifier, pbar=None, loading_bar=False, 
                        terminating_index=np.inf, step=None, 
                        mean=None, std=None, attr_idx=-1, avg=False, mask=None):
        y_gt = []
        y_pred = []
        classifier.eval() 
        
        if loading_bar:
            with tqdm(total=len(eval_loader), desc='Evaluating Test Accuracy', unit='batch') as lbar: 
                y_gt, y_pred = self.evaluation_loop(eval_loader, classifier, y_gt, y_pred, pbar=pbar, mean=mean, 
                                                    lbar=lbar, std=std, mask=mask)
        else:
            y_gt, y_pred = self.evaluation_loop(eval_loader, classifier, y_gt, y_pred, pbar=pbar, mean=mean, 
                                                std=std, mask=mask)  
        y_gt = torch.cat(y_gt, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        if attr_idx >= 0:
            return average_precision_score(y_gt[:,attr_idx], y_pred[:,0])   # pred is only 1 dimensional
        
        acc_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

        for i in range(y_gt.shape[1]):
            # ap = average_precision_score(y_gt[:,i], y_pred[:,i])
            acc = accuracy_score((y_gt[:,i] > 0).astype(int), (y_pred[:,i] > 0.5).astype(int)      )  
            precision = precision_score( (y_gt[:,i] > 0).astype(int), (y_pred[:,i] > 0.5).astype(int)) 
            recall = recall_score(y_gt[:,i] > 0, y_pred[:,i] > 0.5) 
            f1 = f1_score(y_gt[:,i] > 0, y_pred[:,i] > 0.5) 
            auc = roc_auc_score(y_gt[:,i]>0, y_pred[:,i]>0.5) 

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1) 
            auc_list.append(auc) 
            
            if pbar:
                pbar.update(1)  
        
        wandb.log({'acc': np.mean(acc_list), 
                    'precision': np.mean(precision_list), 
                    'recall': np.mean(recall_list), 
                    'f1': np.mean(f1_list), 
                    'auc': np.mean(auc_list)}, 
                    step=step)  
        
        # Flatten the arrays
        y_gt_flat = (y_gt.ravel() > 0).astype(int) 
        y_pred_flat = (y_pred.ravel() > 0.5).astype(int)

        # Calculate metrics globally
        global_acc = accuracy_score(y_gt_flat, y_pred_flat)
        global_precision = precision_score(y_gt_flat, y_pred_flat)
        global_recall = recall_score(y_gt_flat, y_pred_flat)
        global_f1 = f1_score(y_gt_flat, y_pred_flat)
        global_auc = roc_auc_score(y_gt_flat, y_pred_flat) 
        
        # Log the global metrics
        wandb.log({
            'global_acc': global_acc, 
            'global_precision': global_precision, 
            'global_recall': global_recall, 
            'global_f1': global_f1, 
            'global_auc': global_auc
        })
                   
    def evaluation_loop(self, eval_loader, classifier, y_gt, y_pred, pbar=None, lbar=None, mean=None, std=None, 
                        mask=None):
        for batch in eval_loader: 
            image, labels = batch['imgs'].to(self.args.device), batch['labels'].to(self.args.device)  
            latent = self.encoder(image) 
            predictions = classifier(latent)    
            
            y_gt.append(labels.detach().cpu())
            y_pred.append(predictions.detach().cpu()) 
            
            if pbar: 
                pbar.update(1)    

            if lbar:
                lbar.update(1)  
        return y_gt, y_pred 
    

    def save_classifier(self, classifier, type='all', epoch=None): 
        if not os.path.exists(f'runs/m-{self.args.ext}/{type}'):
            os.makedirs(f'runs/m-{self.args.ext}/{type}') 
            
        extra_string = f'_{epoch}' if epoch is not None else '' 
        torch.save(classifier, f'runs/m-{self.args.ext}/{type}/classifier{extra_string}.pt') 


    def eval_accuracy(self, mode=2, batch_size=128):
        test_loader =  DataLoader(dataset = CelebA_Dataset(mode=mode, dataset=self.args.ext) , batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=8, 
                                persistent_workers=True)  
        
    
        classifier = torch.load(f'runs/m-{self.args.ext}/best/classifier.pt')  

        self.eval_multitask(test_loader, classifier=classifier,   
                                      loading_bar=True)  
        

        # wandb.log({"test_ap": test_ap})  

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  
    EvalCeleba_Test(args=args).train()  
    EvalCeleba_Test(args=args).eval_accuracy() 