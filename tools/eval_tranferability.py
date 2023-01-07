from transformers import BertForSequenceClassification, BertTokenizer
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPlugin
import sys
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class MyDataSet(Dataset):
    def __init__(self, sentences, labels):
        super(MyDataSet, self).__init__()
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, index):
        text = self.sentences[index]
        label = self.labels[index]
        result = {'label': label, 'text': text}
        return result

    def __len__(self):
        return len(self.sentences)


class BertEvalution(pl.LightningModule):
    def __init__(self, args):
        super(BertEvalution, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(args.pre_trained_model, num_labels=args.n_classes)
        self.tokenizer = BertTokenizer.from_pretrained(args.pre_trained_model, do_lower_case=True,
                                                       model_max_length=args.max_length)
        self.file_path = args.dataset
        self.batch_size = args.batch_size
        self.dataset = self.init_dataset()
        
    def init_dataset(self):
        f = open(self.file_path, 'r')
        label = f.readline().strip('\n')
        total = 0
        skip = 0
        success = 0
        fail = 0
        sentences = []
        labels = []
        while label:
            total += 1
            origin = f.readline().strip('\n')
            attack = f.readline().strip('\n')
            stat = f.readline().strip('\n')

            if stat == 'fail':
                fail += 1
            if stat == 'success':
                success += 1
            if stat == 'skip':
                skip += 1
           
            if stat == 'success':
                labels.append(int(label))
                labels.append(int(label))
                sentences.append(origin)
                sentences.append(attack)

            label = f.readline().strip('\n')

        output = {
            'total': total,
            'clean_acc': (total - skip) * 1.0 / total,
            'attack_acc': success * 1.0 / (success + fail)
        }
        print(output)
        print(len(labels))
        sys.stdout.flush()

        return MyDataSet(sentences, labels)

    def forward(self, batch):
        labels = batch['label']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        labels = batch['label']
        predictions = torch.argmax(outputs[1], dim=-1).data
        correct_flag = (predictions == labels)
        success = 0
        total = 0
        for i in range(0, len(labels), 2):
            if correct_flag[i]:
                total += 1
                if not correct_flag[i+1]:
                    success += 1
            
        loss = outputs[0]
        return (success, total, len(labels) / 2)
        # return (len(labels)-torch.sum(correct_flag), len(labels))
        # return (torch.sum(correct_flag), len(labels))

    def test_epoch_end(self, outputs):
        ## correct_count, batch_size
        accuracy = sum([out[1] for out in outputs]) * 1.0 / sum(out[2] for out in outputs)
        success = sum([out[0] for out in outputs]) * 1.0 / sum(out[1] for out in outputs)
        output = {"test_acc": accuracy, "test_success": success}
        # output = {"test_success": success}
        # accuracy = sum([out[0] for out in outputs]) * 1.0 / sum(out[1] for out in outputs)
        # output = {"test_acc": accuracy}
        self.log_dict(output, on_epoch=True, prog_bar=True, sync_dist=True)
        print(output)
        return output

    def test_dataloader(self):

        test_dataloader = DataLoader(
            dataset=self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
        return test_dataloader
    
    def train_dataloader(self):

        train_dataloader = DataLoader(
            dataset=self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
        return train_dataloader

    def _collate_fn(self, batch):
        label = torch.tensor([item['label'] for item in batch])
        text = [item['text'] for item in batch]
        outputs = self.tokenizer(text, truncation=True, padding=True)
        input_ids = torch.tensor(outputs["input_ids"])
        attention_mask = torch.tensor(outputs["attention_mask"])
        return {'label': label, 'text': text, 'input_ids': input_ids, 'attention_mask': attention_mask}


if __name__ == '__main__':
    print('PyTorch V ersion {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))

    parser = argparse.ArgumentParser()

    # dataset optionsargs.splits
    data_args = parser.add_argument_group('fine tune options')
    data_args.add_argument('--dataset', type=str, default='ag_news')
    data_args.add_argument('--pre_trained_model', type=str, default='bert-base-uncased')
    data_args.add_argument('--summary_dir', type=str, default='summary_test')
    data_args.add_argument('--n_classes', type=int, default=4)
    data_args.add_argument('--max_length', type=int, default=128)
    data_args.add_argument('--ngpu', type=int, default=2)
    data_args.add_argument('--local_rank', type=int, default=0)
    data_args.add_argument('--batch_size', type=int, default=128)


    args = parser.parse_args()

    trainer = pl.Trainer(
        gpus=args.ngpu,
        accelerator='ddp',
        # plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        plugins='deepspeed',
        precision=16
    )
    
    model = BertEvalution(args)
    trainer.test(model)

