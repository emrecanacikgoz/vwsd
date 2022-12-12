import torch
from torch import optim
import pytorch_lightning as pl


from vwsd.model import CLIPZeroShotBaseline


class BaseExperiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_name = self.config.get('model', {}).get('name', 'UndefinedModel')
        if model_name == 'CLIPZeroShotBaseline':
            self.model = CLIPZeroShotBaseline()
        else:
            raise NotImplementedError(f'There is no such model as {model_name}')
        self.save_hyperparameters(config)

    def forward(self, batch):
        return self.model(**batch)

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