import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

from vwsd.model import CLIPFinetune

class BaseExperiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(f"Config: {config}")
        model_name = self.config.get('model', {}).get('name', 'UndefinedModel')
        if model_name == 'CLIPZeroShotBaseline':
            self.model = CLIPZeroShotBaseline()
        elif model_name == 'CLIPFinetune':
            self.model = CLIPFinetune()
        else:
            raise NotImplementedError(f'There is no such model as {model_name}')
        self.save_hyperparameters(config)

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        labels = batch["labels"]
        labels = labels.squeeze(0)

        loss = F.cross_entropy(output, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        labels = batch["labels"]

        _, preds = torch.max(output.data, 1)
        labels = labels.squeeze(0)
        print(f"preds.shape, labels.shape: {preds.shape, labels.shape}")
        acc = (preds == labels).sum() / labels.shape[0]

        loss = F.cross_entropy(output, labels)

        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        prob = output.softmax(dim=1)
        N = prob.shape[-1]

        preds = prob.topk(N).indices.cpu().tolist()
        image_files = []
        for i, pred in enumerate(preds):
            this = [batch['image_files'][i][j] for j in pred]
            image_files.append(this)
        indexes = batch['indexes']
        return {
            'image_files': image_files,
            'indexes': indexes,
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-7)
        return optimizer
