from torch.distributed import init_process_group
from moco_gready_attack import GreedyAttack
import torch
from utils import BertBase
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl
import argparse
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import moco.builder
import torch.nn as nn
from deepspeed.ops.adam import FusedAdam

def get_inputs(sentences, tokenizer, device):
    outputs = tokenizer(sentences, truncation=True, padding=True)
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


class BertCLClassification(BertBase):
    def __init__(self, args):
        super(BertCLClassification, self).__init__(args)
        self.model = moco.builder.MoCo(pre_trained_path=self.pre_trained_model, hidden_size=768)
        self.criterion = nn.CrossEntropyLoss()
        self.greedy_attack = GreedyAttack(args=args, tokenizer=self.tokenizer, criterion=self.criterion)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        # optimizer = FusedAdam(params=optimizer_grouped_parameters, lr=self.learning_rate)

        num_training_steps = self.epochs * int(len(self.train_dataset) / self.batch_size / self.ngpu)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_proportion * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def forward(self, batch):

        sentences = batch['text']
        input_ids = batch['input_ids']

        adv_words = self.greedy_attack.adv_attack_samples(model=self.model, input_ids=input_ids,
                                                          device=self.device)
        sentences = [' '.join(e) for e in adv_words] + sentences

        input_ids, attention_mask = get_inputs(sentences, self.tokenizer, self.device)
        return input_ids, attention_mask

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask = self(batch)
        torch.cuda.empty_cache()

        output, target = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(output, target)

        self.log_dict({'loss': loss}, on_step=True, prog_bar=True, sync_dist=True)
        print({'loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask = self(batch)
        torch.cuda.empty_cache()

        output, target = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(output, target)

        self.log_dict({'val_loss': loss}, on_step=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):

        input_ids, attention_mask = self(batch)
        torch.cuda.empty_cache()

        output, target = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(output, target)

        self.log_dict({'test_loss': loss}, on_step=True, prog_bar=True, sync_dist=True)

        return loss

if __name__ == '__main__':
    print('PyTorch V ersion {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))

    parser = argparse.ArgumentParser()

    # dataset optionsargs.splits
    data_args = parser.add_argument_group('contrastive learning options')
    data_args.add_argument('--dataset', type=str, default='ag_news')
    data_args.add_argument('--pre_trained_model', type=str, default='bert-base-uncased')
    data_args.add_argument('--summary_dir', type=str, default='summary_adv')
    data_args.add_argument('--n_classes', type=int, default=4)
    data_args.add_argument('--max_length', type=int, default=128)
    data_args.add_argument('--learning_rate', type=float, default=2e-5)
    data_args.add_argument('--warmup_proportion', type=float, default=0.1)
    data_args.add_argument('--ngpu', type=int, default=2)
    data_args.add_argument('--local_rank', type=int, default=0)
    data_args.add_argument('--batch_size', type=int, default=128)
    data_args.add_argument('--epochs', type=int, default=5)
    data_args.add_argument('--n_candidates', type=int, default=25)
    data_args.add_argument('--max_loops', type=int, default=10)
    data_args.add_argument('--splits', type=int, default=128)
    data_args.add_argument('--sim_thred', type=float, default=0.5)
    data_args.add_argument('--embedding_path', type=str, default='dataset/counter-fitted-vectors.txt')
    data_args.add_argument('--sim_path', type=str, default='dataset/cos_sim_counter_fitting.npy')
    data_args.add_argument('--cos_sim', action='store_true')
    data_args.add_argument('--synonym', type=str, default='cos_sim')
    data_args.add_argument('--test', action='store_true')
    data_args.add_argument('--save_model', action='store_true')
    data_args.add_argument('--checkpoint_path', type=str)
    data_args.add_argument('--model_path', type=str)
    data_args.add_argument('--train_size', type=float, default=1.0)
    data_args.add_argument('--val_size', type=float, default=1.0)
    data_args.add_argument('--test_size', type=float, default=1.0)

    args = parser.parse_args()
    model = BertCLClassification(args)
    # model.train()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir=args.summary_dir,
        name=args.dataset
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor='val_loss')

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.ngpu,
        check_val_every_n_epoch=3,
        accelerator='ddp',
        max_epochs=args.epochs,
        callbacks=[lr_monitor, checkpoint_callback]
        # plugins='deepspeed',
        # precision=16
    )

    if args.save_model:
        model = BertCLClassification.load_from_checkpoint(args.checkpoint_path, args=args)
        model.model.encoder_q.save_pretrained(args.model_path)
        model.tokenizer.save_pretrained(args.model_path)
        exit(0)

    if args.test:
        trainer.test(model)
        exit(0)

    trainer.fit(model)

