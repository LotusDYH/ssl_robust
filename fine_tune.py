from transformers import BertForSequenceClassification
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import BertBase


class BertClassification(BertBase):
    def __init__(self, args):
        super(BertClassification, self).__init__(args)
        self.model = BertForSequenceClassification.from_pretrained(self.pre_trained_model, num_labels=self.n_classes)

    def forward(self, batch):
        labels = batch['label']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]

        self.log_dict({"loss": loss}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        labels = batch['label']
        predictions = torch.argmax(outputs[1], dim=-1).data
        correct_count = (predictions == labels).float().sum()
        loss = outputs[0]

        self.log_dict({'val_loss': loss}, prog_bar=True, sync_dist=True)
        return (correct_count, len(labels))

    def validation_epoch_end(self, outputs):
        ## correct_count, batch_size
        accuracy = sum([out[0] for out in outputs]) * 1.0 / sum(out[1] for out in outputs)
        output = {"val_acc": accuracy}
        self.log_dict(output, prog_bar=True, sync_dist=True)
        return output

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        labels = batch['label']
        predictions = torch.argmax(outputs[1], dim=-1).data
        correct_count = (predictions == labels).float().sum()
        loss = outputs[0]

        self.log_dict({'test_loss': loss}, prog_bar=True, sync_dist=True)
        return (correct_count, len(labels))

    def test_epoch_end(self, outputs):
        ## correct_count, batch_size
        accuracy = sum([out[0] for out in outputs]) * 1.0 / sum(out[1] for out in outputs)
        output = {"test_acc": accuracy}
        self.log_dict(output, prog_bar=True, sync_dist=True)
        return output


if __name__ == '__main__':
    print('PyTorch V ersion {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))

    parser = argparse.ArgumentParser()

    # dataset optionsargs.splits
    data_args = parser.add_argument_group('fine tune options')
    data_args.add_argument('--dataset', type=str, default='ag_news')
    data_args.add_argument('--pre_trained_model', type=str, default='bert-base-uncased')
    data_args.add_argument('--summary_dir', type=str, default='summary')
    data_args.add_argument('--n_classes', type=int, default=4)
    data_args.add_argument('--max_length', type=int, default=128)
    data_args.add_argument('--learning_rate', type=float, default=2e-5)
    data_args.add_argument('--train_size', type=float, default=1.0)
    data_args.add_argument('--val_size', type=float, default=1.0)
    data_args.add_argument('--test_size', type=float, default=1.0)
    data_args.add_argument('--warmup_proportion', type=float, default=0.1)
    data_args.add_argument('--ngpu', type=int, default=2)
    data_args.add_argument('--local_rank', type=int, default=0)
    data_args.add_argument('--batch_size', type=int, default=128)
    data_args.add_argument('--epochs', type=int, default=5)
    data_args.add_argument('--test', action='store_true')
    data_args.add_argument('--save_model', action='store_true')
    data_args.add_argument('--checkpoint_path', type=str)
    data_args.add_argument('--model_path', type=str)

    args = parser.parse_args()

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir=args.summary_dir,
        name=args.dataset
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{step}-{val_acc:.3f}-{val_loss:.3f}',
        mode='min',
        save_last=True,
        save_top_k=-1
    )

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.ngpu,
        check_val_every_n_epoch=2,
        # val_check_interval=1,
        accelerator='ddp',
        max_epochs=args.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        # plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        plugins='deepspeed',
        precision=16
    )

    if args.save_model:
        model = BertClassification.load_from_checkpoint(args.checkpoint_path, args=args)
        model.model.save_pretrained(args.model_path)
        model.tokenizer.save_pretrained(args.model_path)
        print('use checkpoint {}, save the model to {}'.format(args.checkpoint_path, args.model_path))
        exit(0)

    model = BertClassification(args)

    if args.test:
        trainer.test(model)
        exit(0)

    trainer.fit(model)
