from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel, AdamW, BertConfig

import pandas as pd
import numpy as np
import os
import time
import math
import re

from random import *
from datetime import date
from collections import Counter, defaultdict


class ClassifierBert(nn.Module):
    def __init__(self, device, tasks=["toxic"], labels=2):
        super(ClassifierBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased",
                                              return_dict=True)

        self.tasks = tasks
        self.labels = labels
        self.linear_layer = dict()
        for task in tasks:
            self.linear_layer[task] = nn.Linear(BertConfig().hidden_size, labels).to(device)

    def forward(self, ids, attn):
        outputs = self.bert(
            ids,
            attn
        )
        self.task_logits = dict()
        hidden = outputs.last_hidden_state[:, 0, :]
        for task in self.tasks:
            self.task_logits[task] = self.linear_layer[task](hidden)
        return self.extract_outputs()

    def extract_outputs(self):
        if self.labels > 2:
            logits = {str(label): self.task_logits["toxic"][:, label] for label in range(self.labels)}
            sig_logits = {str(label): torch.sigmoid(logits[str(label)]) for label in range(self.labels)}
            predictions = {str(label): [1 if x > 0.5 else 0 for x in sig_logits[str(label)]] for label in
                           range(self.labels)}
        else:
            logits = self.task_logits
            predictions = {task: [x.item() for x in torch.argmax(logits[task], dim=-1)] for task in self.tasks}
        return logits, predictions


"""## General classifier
The class that performs multi-task, multi-label classifications for a given dataset
"""


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


class ToxicityClassifier():
    def __init__(self, data, annotators, params):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda')
        self.data = data
        self.annotators = annotators

        self.multi_label, self.multi_task, self.ensemble, self.single = False, False, False, False
        setattr(self, params.task, True)

        if self.single:
            self.task_labels = ["toxic"]
        else:
            self.task_labels = annotators

        self.majority_vote()
        self.certainty()
        print("Data shape after majority voting", self.data.shape)

        # Setting the parameters
        self.params = params
        print([(k, v) for k, v in self.params.__dict__.items()])

    def majority_vote(self):
        self.data["toxic"] = (self.data[self.annotators].sum(axis=1) / \
                              self.data[self.annotators].count(axis=1) >= 0.5).astype(int)
        print(sum(self.data["toxic"]))

    def certainty(self):
        self.data["certainty"] = 1 - (self.data[self.annotators].sum(axis=1) \
                                      * (self.data[self.annotators].count(axis=1) - self.data[self.annotators].sum(
                    axis=1)) \
                                      / (self.data[self.annotators].count(axis=1) * self.data[self.annotators].count(
                    axis=1)))

    def CV(self):
        if self.ensemble:
            ensemble_results = pd.DataFrame()
            for annotator in self.annotators:
                print("Training model for annotator", annotator)
                # self.data = self.data.rename(columns = {annotator: "toxic", "toxic": "_toxic"})
                self.task_labels = ["toxic"]
                # annotator_data = self.data.dropna(subset=[annotator])[["text", annotator]]
                scores, results = self._CV(self.data.rename(columns={annotator: "toxic", "toxic": "_toxic"}))
                ensemble_results[annotator + "_pred"] = results["toxic_pred"]
                ensemble_results[annotator + "_label"] = results["toxic_label"]
                ensemble_results[annotator + "_masked_pred"] = results["toxic_masked_pred"]
                ensemble_results[annotator + "_masked_label"] = results["toxic_masked_label"]
                # self.data = self.data.rename(columns = {"toxic": annotator, "_toxic": "toxic"})
            self.task_labels = self.annotators
            scores = self.report_results(ensemble_results)
            return scores, ensemble_results
        else:
            return self._CV(self.data)

    def masks(self, df):
        df = df.replace(0, 1)
        df = df.replace(np.nan, 0)
        new_labels = LabelEncoder().fit_transform([''.join(str(l) for l in row) for i, row in df.iterrows()])
        return new_labels

    def _CV(self, data):
        if self.params.stratified:
            kfold = StratifiedKFold(n_splits=self.params.num_folds, shuffle=True,
                                    random_state=self.params.random_state)
        else:
            kfold = KFold(n_splits=self.params.num_folds, shuffle=True, random_state=self.params.random_state)

        results = pd.DataFrame()
        i = 1
        test_results = defaultdict(list)
        for train_idx, test_idx in kfold.split(np.zeros(self.data.shape[0]), self.masks(self.data[self.annotators])):
            print("Fold #", i)

            train = data.loc[train_idx].reset_index()
            test = data.loc[test_idx].reset_index()

            train_batches = self.get_batches(train)
            test_batches = self.get_batches(test)

            self.train_model(train_batches)

            # testing on the validation set
            fold_result = self.predict(test_batches)
            print("Test:")
            print(self.report_results(fold_result))

            fold_result["fold"] = pd.Series([i for id in test_idx])
            results = results.append(fold_result)
            i += 1

        scores = self.report_results(results)
        print(scores)
        return scores, results

    def new_model(self):
        if self.multi_task:
            return ClassifierBert(self.device, tasks=self.annotators)
        elif self.multi_label:
            return ClassifierBert(self.device, labels=len(self.annotators))
        else:
            return ClassifierBert(self.device)

    def create_loss_functions(self):
        losses = dict()
        #self.class_weight = dict()

        for task_label in self.task_labels:
            _labels = [int(x) for x in self.data[task_label].dropna().tolist()]
            weight = compute_class_weight('balanced',
                                          np.unique(_labels),
                                          _labels)
            if len(weight) == 1:
                weight = [0.01, 1]
            weight = torch.tensor(weight, dtype=torch.float32).to(self.device)

            if self.multi_label:
                ### multi-label
                losses[task_label] = nn.BCEWithLogitsLoss(reduction="sum")  # , pos_weight=class_weight)
            else:
                losses[task_label] = nn.CrossEntropyLoss(weight=weight)

        return losses

    def train_model(self, batches):
        self.model = self.new_model()
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.learning_rate)

        train_batches, val_batches = train_test_split(batches,
                                                      shuffle=True,
                                                      random_state=self.params.random_state,
                                                      test_size=.1)

        self.loss = self.create_loss_functions()
        for epoch in range(self.params.num_epochs):

            loss_val = 0
            self.model.train()

            for batch in train_batches:
                X_ids = torch.tensor(batch["inputs"]).to(self.device)
                X_att = torch.tensor(batch["attentions"]).to(self.device)
                if len([x for task_label in self.task_labels for x in batch["masks"][task_label]]) == 0:
                    continue

                logits, _ = self.model(X_ids, attn=X_att)
                class_loss = dict()
                weighted_sum = 0
                for task_label in self.task_labels:
                    masked_logits = logits[task_label][batch["masks"][task_label]]
                    masked_labels = [batch["labels"][task_label][x] for x in batch["masks"][task_label]]
                    if self.multi_task or self.ensemble:
                        masked_labels = torch.tensor(masked_labels).type("torch.LongTensor").to(self.device)
                    else:
                        masked_labels = torch.tensor(masked_labels).to(self.device)

                    if len(batch["masks"][task_label]) > 0:
                        ## list of loss values for each batch instance
                        class_loss[task_label] = self.loss[task_label](masked_logits, masked_labels)

                        ## using a column of the data as the weight for loss value of each instance
                        # Batch["weight"] shows the instance weight (based on its certainty), class_weight shows the class weight for positive and negative labels
                        # batch["weights"][batch_i] *
                        """
                        class_loss[task_label] = sum([ batch_loss[mask_i] * self.class_weight[task_label][masked_labels[mask_i]]
                                                      for mask_i, batch_i in enumerate(batch["masks"][task_label])])
                        weighted_sum += sum([self.class_weight[task_label][label] for label in masked_labels])
                        """
                total_loss = sum(class_loss.values())  # / weighted_sum
                loss_val += total_loss.item()
                total_loss.backward()
                self.optimizer.step()

            print("Epoch", epoch, "-", "Loss", round(loss_val, 3))
            if val_batches:
                val_results = self.predict(val_batches, self.model)
                print("Validation:")
                print(self.report_results(val_results))

    def predict(self, batches, model=None):
        self.model.eval()
        results = defaultdict(list)

        for batch in batches:

            X_ids = torch.tensor(batch["inputs"]).to(self.device)
            X_att = torch.tensor(batch["attentions"]).to(self.device)

            _, predictions = self.model(X_ids, attn=X_att)

            for task_label in self.task_labels:
                masked_labels = [x if x in batch["masks"][task_label] else np.nan for x in batch["labels"][task_label]]
                masked_predictions = [x if x in batch["masks"][task_label] else np.nan for x in predictions[task_label]]

                results[task_label + "_masked_pred"].extend(masked_predictions)
                results[task_label + "_masked_label"].extend(masked_labels)
                results[task_label + "_pred"].extend(predictions[task_label])
                results[task_label + "_label"].extend(batch["labels"][task_label])

        return pd.DataFrame.from_dict(results)

    def report_results(self, results):
        if len(self.task_labels) > 1:
            label_cols = [col + "_label" for col in self.annotators]
            pred_cols = [col + "_pred" for col in self.annotators]

            masked_label_cols = [col + "_masked_label" for col in self.annotators]
            masked_pred_cols = [col + "_masked_pred" for col in self.annotators]

            toxic_label = results[label_cols].sum(axis=1) / results[label_cols].count(axis=1) >= 0.5
            toxic_pred = results[pred_cols].sum(axis=1) / results[pred_cols].count(axis=1) >= 0.5

            masked_toxic_label = results[masked_label_cols].sum(axis=1) / results[masked_label_cols].count(
                axis=1) >= 0.5
            masked_toxic_pred = results[masked_pred_cols].sum(axis=1) / results[masked_pred_cols].count(axis=1) >= 0.5

            print("Accuracy of the majority vote (after masking):")

            result_cat = masked_toxic_label.map({True: "T", False: "F"}) + masked_toxic_pred.map(
                {True: "T", False: "F"})
            result_cat = result_cat.map({"TT": "TP", "FF": "TN", "TF": "FN", "FT": "FP"})
            true_results = result_cat.isin(["TP", "TN"])

            counts = Counter(result_cat)
            a = Counter(true_results)[True] / results.shape[0]
            p = counts["TP"] / max((counts["TP"] + counts["FP"]), 1)
            r = counts["TP"] / max((counts["TP"] + counts["FN"]), 1)
            try:
                f = 2 * p * r / (p + r)
            except Exception:
                f = 0
            print({"A": round(a, 3),
                   "P": round(p, 3),
                   "R": round(r, 3),
                   "F1": round(f, 3)})

            print("Accuracy of the majority vote (using all annotator heads):")
        else:
            toxic_label = results["toxic_label"] == 1
            toxic_pred = results["toxic_pred"] == 1
            print("Accuracy of single label")

        result_cat = toxic_label.map({True: "T", False: "F"}) + toxic_pred.map({True: "T", False: "F"})
        result_cat = result_cat.map({"TT": "TP", "FF": "TN", "TF": "FN", "FT": "FP"})
        true_results = result_cat.isin(["TP", "TN"])

        counts = Counter(result_cat)
        a = Counter(true_results)[True] / results.shape[0]
        p = counts["TP"] / max((counts["TP"] + counts["FP"]), 1)
        r = counts["TP"] / max((counts["TP"] + counts["FN"]), 1)
        try:
            f = 2 * p * r / (p + r)
        except Exception:
            f = 0

        scores = {"A": round(a, 3),
                  "P": round(p, 3),
                  "R": round(r, 3),
                  "F1": round(f, 3)}
        return scores

    def get_batches(self, data):
        if isinstance(self.params.sort_by, str):
            data = data.sort_values(by=[self.params.sort_by], ascending=False).reset_index()
        batches = list()

        for s in range(0, len(data), self.params.batch_size):
            e = s + self.params.batch_size if s + self.params.batch_size < len(data) else len(data)
            data_info = self.batch_to_info(data["text"].tolist()[s: e])

            anno_batch = dict()
            mask_batch = dict()
            for task_label in self.task_labels:
                anno_batch[task_label] = data[task_label].tolist()[s: e]
                mask_batch[task_label] = [i for i, h in enumerate(anno_batch[task_label]) \
                                          if not math.isnan(h)]
            data_info["labels"] = anno_batch
            data_info["masks"] = mask_batch

            data_info["majority_vote"] = data["toxic"].tolist()[s: e]
            data_info["batch_len"] = e - s
            if isinstance(self.params.batch_weight, str):
                data_info["weights"] = data[self.params.batch_weight].tolist()[s: e]
            else:
                data_info["weights"] = [1 for i in range(e - s)]
            batches.append(data_info)
        return batches

    def batch_to_info(self, batch):
        batch_info = dict()
        if isinstance(self.params.max_len, int):
            tokens = self.tokenizer(batch,
                                    padding="max_length",
                                    max_length=self.params.max_len,
                                    truncation=True)
        else:
            tokens = self.tokenizer(batch,
                                    padding=True,
                                    truncation=True)
        batch_info["inputs"] = tokens["input_ids"]
        batch_info["attentions"] = tokens["attention_mask"]
        return batch_info