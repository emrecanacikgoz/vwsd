import os
import os.path as osp

import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from vwsd.experiment import BaseExperiment as Experiment
from vwsd.data import VWSDDataModule as DataModule
from vwsd.util import process_path, write_results
from pytorch_lightning.callbacks import ModelCheckpoint
import faulthandler


CONFIG_DIR = osp.abspath(osp.join(__file__, "../..", "config"))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="clip_ft")
def main(config):
    faulthandler.enable()
    config = OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)
    assert config['output'] is not None
    pl.seed_everything(config["seed"])
    print(config)
    config['output'] = process_path(config['output'])
    print(config["model"])
    load = config["model"]["load"]

    model_name = 'openai/clip-vit-large-patch14'
    transform = CLIPFeatureExtractor.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    #checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/',filename='{epoch}-{val_loss:.3f}', every_n_epochs=1)
    
    trainer = pl.Trainer(
        logger=None,
        callbacks=None,#checkpoint_callback,
        **config["trainer"])

    if not load:
        train_dm = DataModule(transform=transform, tokenizer=tokenizer, **config['data'])
        train_dm.setup(stage='fit')
        experiment = Experiment(config)

        results = trainer.fit(experiment, datamodule=train_dm)

        trial_dm = DataModule(transform=transform, tokenizer=tokenizer, **config['data'])
        trial_dm.setup(stage='predict')

        results = trainer.predict(experiment, datamodule=trial_dm)
        write_results(results, config['output'])
    else:
        experiment = Experiment.load_from_checkpoint("/kuacc/users/oince22/vwsd/scripts/checkpoints/epoch=9-step=28960.ckpt")
        write_results(results, config['output'])


if __name__ == "__main__":
    main()
