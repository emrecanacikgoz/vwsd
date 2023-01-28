import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl

from vwsd.model import CLIPFinetune

class BaseExperiment(pl.LightningModule):
    def __init__(self, config, lr=5e-7):
        super().__init__()
        self.config = config
        print(f"Config: {config}")
        model_name = self.config.get('model', {}).get('name', 'UndefinedModel')
        self.lr = self.config.get('model', {}).get('lr', 5e-6)
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


    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_lbfgs=False,
        using_native_amp=None
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < 5000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
