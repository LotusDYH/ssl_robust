from contrastive_greedy_attack import GreedyAttack
import torch
from utils import BertBase
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl
import argparse
from transformers import BertModel
from my_model import CLModel
from my_loss import pairwise_similarity, get_loss
from torch.utils.data import DataLoader
from deepspeed.ops.adam import FusedAdam
from transformers.optimization import get_linear_schedule_with_warmup


def get_inputs(sentences, tokenizer, device):
    outputs = tokenizer(sentences, truncation=True, padding=True)
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


class BertCLClassification(BertBase):
    def __init__(self, args):
        super(BertCLClassification, self).__init__(args)
        self.model = BertModel.from_pretrained(self.pre_trained_model)
        self.model_g = CLModel(hidden_size=768)
        self.greedy_attack = GreedyAttack(args=args, tokenizer=self.tokenizer)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)] +
                          [p for n, p in self.model_g.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)] +
                          [p for n, p in self.model_g.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = FusedAdam(params=optimizer_grouped_parameters, lr=self.learning_rate)
        ## optimizer = DeepSpeedCPUAdam(model_params=optimizer_grouped_parameters, lr=self.learning_rate)

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

        adv_words = self.greedy_attack.adv_attack_samples(model=self.model, model_g=self.model_g, input_ids=input_ids,
                                                          device=self.device)
        sentences = sentences + [' '.join(e) for e in adv_words]

        input_ids, attention_mask = get_inputs(sentences, self.tokenizer, self.device)
        return input_ids, attention_mask

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask = self(batch)
        torch.cuda.empty_cache()

        bert_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        outputs_z = self.model_g(bert_outputs[1])
        similarity_matrix = pairwise_similarity(outputs=outputs_z, ngpu=self.ngpu)
        loss = get_loss(similarity_matrix=similarity_matrix, device=self.device)

        self.log_dict({'loss': loss}, on_step=True, prog_bar=True, sync_dist=True)
        print({'loss': loss})

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask = self(batch)
        torch.cuda.empty_cache()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        outputs_z = self.model_g(outputs[1])
        similarity_matrix = pairwise_similarity(outputs=outputs_z, ngpu=self.ngpu)
        loss = get_loss(similarity_matrix=similarity_matrix, device=self.device)

        self.log_dict({'val_loss': loss}, on_step=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):

        input_ids, attention_mask = self(batch)
        torch.cuda.empty_cache()

        words = [self.tokenizer.decode(ids, skip_special_tokens=True,
                          clean_up_tokenization_spaces=False).split(" ") for ids in input_ids]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        outputs_z = self.model_g(outputs[1])
        similarity_matrix = pairwise_similarity(outputs=outputs_z, ngpu=self.ngpu)
        loss = get_loss(similarity_matrix=similarity_matrix, device=self.device)

        ### output attack samples
        # for i in range(len(words) / 2):
        #     a = words[i]
        #     b = words[i + len(words) / 2]
        #     strmy = ""
        #     for j in range(len(a)):
        #         if a[j] == b[j]:
        #             strmy = strmy + a[j] + " "
        #         else:
        #             strmy = strmy + "!!!!!" + a[j] + "->" + b[j] + "!!!!!"

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
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir=args.summary_dir,
        name=args.dataset
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor='val_loss')

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.ngpu,
        check_val_every_n_epoch=1,
        accelerator='ddp',
        max_epochs=args.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        # plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        plugins='deepspeed',
        precision=16
    )

    if args.save_model:
        model = BertCLClassification.load_from_checkpoint(args.checkpoint_path, args=args)
        model.model.save_pretrained(args.model_path)
        model.tokenizer.save_pretrained(args.model_path)
        exit(0)

    if args.test:
        trainer.test(model)
        exit(0)

    trainer.fit(model)
