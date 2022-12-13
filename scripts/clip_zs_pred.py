import os
import os.path as osp

import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from vwsd.experiment import BaseExperiment as Experiment
from vwsd.data import VWSDDataModule as DataModule
from vwsd.util import process_path, write_results


CONFIG_DIR = osp.abspath(osp.join(__file__, "../..", "config"))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="clip_zs")
def main(config):
    config = OmegaConf.to_container(config)
    config = pl.utilities.parsing.AttributeDict(config)
    assert config['output'] is not None
    pl.seed_everything(config["seed"])
    print(config)
    config['model'] = {'name': 'CLIPZeroShotBaseline'}
    config['output'] = process_path(config['output'])

    model_name = 'openai/clip-vit-large-patch14'
    transform = CLIPFeatureExtractor.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
    
    dm = DataModule(transform=transform, tokenizer=tokenizer, **config['data'])
    dm.setup(stage='predict')

    experiment = Experiment(config)
    trainer = pl.Trainer(
        logger=None,
        callbacks=None,
        **config["trainer"])
    results = trainer.predict(experiment, datamodule=dm)
    write_results(results, config['output'])


if __name__ == "__main__":
    main()