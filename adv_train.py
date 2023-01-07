from greedy_attack import GreedyAttack
import torch
from utils import BertBase
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl
import argparse
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader


def get_inputs(sentences, tokenizer, device):
    outputs = tokenizer(sentences, truncation=True, padding=True)
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    return torch.tensor(input_ids).to(device), torch.tensor(attention_mask).to(device)


class BertAdvClassification(BertBase):
    def __init__(self, args):
        super(BertAdvClassification, self).__init__(args)
        self.model = BertForSequenceClassification.from_pretrained(self.pre_trained_model, num_labels=self.n_classes)
        self.greedy_attack = GreedyAttack(args=args, tokenizer=self.tokenizer)
        self.out = None
        if args.output is not None:
            self.out = open(args.output, 'w')

    def forward(self, batch):

        labels = batch['label']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        sentences = batch['text']

        lengths = torch.tensor([len(self.tokenizer.decode(
            ids, skip_special_tokens=True).split(" ")) for ids in input_ids])

        batch_size = len(labels)

        results = {'n_changed': 0.0, 'replace_rate': 0.0, 'n_changed_words': 0.0, 'original_corrects': 0.0}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)

        predictions = torch.argmax(outputs[1], dim=-1).data.cpu()
        one_hot_mask = torch.arange(self.n_classes).unsqueeze(0).repeat(batch_size, 1)
        one_hot_mask = (one_hot_mask == predictions.unsqueeze(1))
        original_probs = torch.softmax(outputs[1], dim=-1).data.cpu()
        probs = torch.masked_select(original_probs, one_hot_mask)

        correct_mask = (predictions == labels.cpu())
        original_corrects = float(torch.sum(correct_mask).item())

        ## only attack correctly classified sentences
        if original_corrects != 0:
            c_labels = torch.masked_select(labels, correct_mask)
            c_lengths = torch.masked_select(lengths, correct_mask)
            c_input_ids = torch.masked_select(input_ids, correct_mask.view(-1, 1)).view(len(c_labels), -1)
            c_attention_mask = torch.masked_select(attention_mask, correct_mask.view(-1, 1)).view(len(c_labels), -1)
            c_predictions = torch.masked_select(predictions, correct_mask)
            c_probs = torch.masked_select(probs, correct_mask)

            perturbed_predictions, attack_words, original_words, intermediate_pred_probs = \
                self.greedy_attack.adv_attack_samples(model=self.model, labels=c_labels,
                                                      input_ids=c_input_ids,
                                                      attention_mask=c_attention_mask,
                                                      original_predictions=c_predictions,
                                                      original_probs=c_probs,
                                                      device=self.device)

            n_changed = float(torch.sum(perturbed_predictions != c_predictions).item())

            n_changed_words = []
            for i in range(len(attack_words)):
                num = 0
                for j in range(len(attack_words[i])):
                    num = num + (attack_words[i][j] != original_words[i][j])
                n_changed_words.append(num)
            n_changed_words = torch.tensor(n_changed_words)

            replace_rate = n_changed_words * 1.0 / c_lengths
            change_labels = perturbed_predictions != c_predictions
            replace_rate *= change_labels
            n_changed_words *= change_labels

            results['n_changed'] = n_changed
            results['original_corrects'] = original_corrects
            results['replace_rate'] = float(torch.sum(replace_rate))
            results['n_changed_words'] = float(torch.sum(n_changed_words))

            ### only use attack samples which fooled the model successfully
            # if n_changed != 0:
            #     train_words = [attack_words[i] for i in range(len(change_labels)) if change_labels[i]]
            #     train_labels = [c_labels[i] for i in range(len(change_labels)) if change_labels[i]]
            #     sentences = sentences + [' '.join(e) for e in train_words]
            #     labels = torch.cat((labels, torch.tensor(train_labels).to(self.device)))

            ## use all attack samples
            train_words = [attack_words[i] for i in range(len(change_labels))]
            train_labels = [c_labels[i] for i in range(len(change_labels))]
            sentences = sentences + [' '.join(e) for e in train_words]
            labels = torch.cat((labels, torch.tensor(train_labels).to(self.device)))

        input_ids, attention_mask = get_inputs(sentences, self.tokenizer, self.device)

        ## output attack samples (correct label, original sample, attack sample, status)
        if self.out != None:
            count = 0
            for i in range(batch_size):
                self.out.write(str(int(batch['label'][i])) + '\n')
                self.out.write(batch['text'][i] + '\n')
                if correct_mask[i]:
                    self.out.write(' '.join(attack_words[count]) + '\n')
                else:
                    self.out.write(batch['text'][i] + '\n')
                if correct_mask[i]:
                    if change_labels[count]:
                        self.out.write('success\n')
                    else:
                        self.out.write('fail\n')
                else:
                    self.out.write('skip\n')
                if correct_mask[i]:
                    count += 1
            self.out.flush()

        return input_ids, attention_mask, labels, results

    def training_step(self, batch, batch_idx):

        batch_size = len(batch['label'])
        ## 0.5 epoch warm up (train on bert-base-uncased)
        ## it is not necessary if train on a fine-tuned model
        if self.global_step <= int(len(self.train_dataset) / self.batch_size / self.ngpu * 0.5):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
        else:
            input_ids, attention_mask, labels, results = self(batch)

        self.model.train()
        torch.cuda.empty_cache()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        predictions = torch.argmax(outputs[1], dim=-1).data
        loss = outputs[0]

        correct_count = float(torch.sum(predictions[:batch_size] == labels[:batch_size]))

        try:
            log_output = {
                'loss': loss,
                'change_rate': results['n_changed'] * 1.0 / results['original_corrects'],
                'replace_rate': results['replace_rate'] * 1.0 / results['n_changed'],
                'changed_words': results['n_changed_words'] * 1.0 / results['n_changed'],
                'acc': correct_count * 1.0 / batch_size
            }
        except:
            log_output = {
                'loss': loss,
                'change_rate': 0,
                'replace_rate': 0,
                'changed_words': 0,
                'acc': correct_count * 1.0 / batch_size
            }
        self.log_dict({'loss': loss}, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        batch_size = len(batch['label'])

        input_ids, attention_mask, labels, results = self(batch)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        predictions = torch.argmax(outputs[1], dim=-1).data
        loss = outputs[0]

        correct_count = float(torch.sum(predictions[:batch_size] == labels[:batch_size]))

        results['correct_count'] = correct_count
        results['batch_size'] = batch_size

        try:
            log_output = {
                'val_loss': loss,
                'val_change_rate': results['n_changed'] * 1.0 / results['original_corrects'],
                'val_replace_rate': results['replace_rate'] * 1.0 / results['n_changed'],
                'val_changed_words': results['n_changed_words'] * 1.0 / results['n_changed'],
                'val_acc': correct_count * 1.0 / batch_size
            }
        except:
            log_output = {
                'val_loss': loss,
                'val_change_rate': 0,
                'val_replace_rate': 0,
                'val_changed_words': 0,
                'val_acc': correct_count * 1.0 / batch_size
            }
        self.log_dict({'val_loss': loss}, prog_bar=True, sync_dist=True)
        print(log_output)
        return (results['n_changed'], results['replace_rate'], results['n_changed_words'], results['original_corrects'],
                results['correct_count'], results['batch_size'])

    def validation_epoch_end(self, outputs):
        ## n_changed, replace_rate, n_changed_words, original_corrects, correct_count, batch_size
        accuracy = sum([out[4] for out in outputs]) * 1.0 / sum(out[5] for out in outputs)
        n_changed = sum([out[0] for out in outputs]) * 1.0
        replace_rate = sum([out[1] for out in outputs]) * 1.0
        n_changed_words = sum([out[2] for out in outputs]) * 1.0
        original_corrects = sum([out[3] for out in outputs]) * 1.0

        try:
            output = {
                "val_acc": accuracy,
                'val_change_rate': n_changed / original_corrects,
                'val_changed_words': n_changed_words / n_changed,
                'val_replace_rate': replace_rate / n_changed
            }
        except:
            output = {
                "val_acc": accuracy,
                'val_change_rate': 0,
                'val_changed_words': 0,
                'val_replace_rate': 0
            }
        self.log_dict(output, prog_bar=True, sync_dist=True)
        return output

    def test_step(self, batch, batch_idx):

        batch_size = len(batch['label'])

        input_ids, attention_mask, labels, results = self(batch)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        predictions = torch.argmax(outputs[1], dim=-1).data

        correct_count = float(torch.sum(predictions[:batch_size] == labels[:batch_size]))

        results['correct_count'] = correct_count
        results['batch_size'] = batch_size

        try:
            log_output = {
                'test_change_rate': results['n_changed'] * 1.0 / results['original_corrects'],
                'test_replace_rate': results['replace_rate'] * 1.0 / results['n_changed'],
                'test_changed_words': results['n_changed_words'] * 1.0 / results['n_changed'],
                'test_acc': correct_count * 1.0 / batch_size
            }
            self.log_dict(log_output, prog_bar=True, sync_dist=True)
        except:
            print('n_changed = 0')

        return (results['n_changed'], results['replace_rate'], results['n_changed_words'], results['original_corrects'],
                results['correct_count'], results['batch_size'])

    def test_epoch_end(self, outputs):
        ## n_changed, replace_rate, n_changed_words, original_corrects, correct_count, batch_size
        accuracy = sum([out[4] for out in outputs]) * 1.0 / sum(out[5] for out in outputs)
        n_changed = sum([out[0] for out in outputs]) * 1.0
        replace_rate = sum([out[1] for out in outputs]) * 1.0
        n_changed_words = sum([out[2] for out in outputs]) * 1.0
        original_corrects = sum([out[3] for out in outputs]) * 1.0

        output = {
            "test_acc": accuracy,
            'test_change_rate': n_changed / original_corrects,
            'test_changed_words': n_changed_words / n_changed,
            'test_replace_rate': replace_rate / n_changed
        }
        self.log_dict(output, prog_bar=True, sync_dist=True)
        if self.out != None:
            self.out.close()
        return output


if __name__ == '__main__':
    print('PyTorch V ersion {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))

    parser = argparse.ArgumentParser()

    # dataset optionsargs.splits
    data_args = parser.add_argument_group('adv train options')
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
    data_args.add_argument('--max_loops', type=int, default=50)
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
    data_args.add_argument('--output', type=str)
    data_args.add_argument('--train_size', type=float, default=1.0)
    data_args.add_argument('--val_size', type=float, default=1.0)
    data_args.add_argument('--test_size', type=float, default=1.0)

    args = parser.parse_args()
    model = BertAdvClassification(args)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(
        save_dir=args.summary_dir,
        name=args.dataset
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=-1, save_last=True, monitor='val_loss')

    trainer = pl.Trainer(
        logger=logger,
        gpus=args.ngpu,
        val_check_interval=0.5,
        accelerator='ddp',
        max_epochs=args.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        # plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, cpu_offload_params=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        plugins='deepspeed',
        precision=16
    )

    if args.save_model:
        model = BertAdvClassification.load_from_checkpoint(args.checkpoint_path, args=args)
        model.model.save_pretrained(args.model_path)
        model.tokenizer.save_pretrained(args.model_path)
        exit(0)

    if args.test:
        trainer.test(model)
        exit(0)

    trainer.fit(model)
