from classifier import *
from params import *

import os
import argparse
from datetime import datetime
import pytz
import re


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The source directory (the root of this repo). It should contain a data folder which "
                             "contains the GHC/ghc_multi.csv file"
                             "and a results folder to save classification scores and predictions.")
    parser.add_argument("--task",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the model to be trained from the list: single, "
                             "multi_task, multi_label, ensemble.")
    parser.add_argument("--max_len",
                        type=int,
                        help="Maximum input length after tokenizing")
    parser.add_argument("--batch_size",
                        type=int)
    parser.add_argument("--learning_rate",
                        type=float)
    parser.add_argument("--num_epochs",
                        type=int)
    parser.add_argument("--random_state",
                        type=int)
    parser.add_argument("--num_folds",
                        type=int)
    parser.add_argument("--stratified",
                        type=bool)
    parser.add_argument("--sort_by",
                        type=str)
    parser.add_argument("--predict",
                        type=str)

    args = parser.parse_args()

    def_params = params()
    def_params.update(args)

    if def_params.task not in ["single", "multi_task", "multi_label", "ensemble"]:
        print("Wrong task input")
        exit(1)

    df = pd.read_csv(os.path.join(def_params.source_dir, "data", "GHC", "ghc_multi.csv"))
    annotators = [col for col in df
                  .columns \
                if re.fullmatch(re.compile(r"[0-9]+"), col)]

    model = ToxicityClassifier(df, annotators=annotators, params=def_params)

    score, results = model.CV()

    score["params"] = ", ".join(key + ": " + str(val) for key, val in def_params.__dict__.items())
    score["task"] = model.params.task

    pacific = pytz.timezone('US/Pacific')
    sa_time = datetime.now(pacific)
    name_time = sa_time.strftime('%m%d%y-%H:%M')
    score["time"] = name_time
    print(score)

    score_dir = os.path.join(def_params.source_dir, "results", "GHC", "classification.csv")
    result_dir = os.path.join(def_params.source_dir, "results", "GHC", name_time + "_" + def_params.task + ".csv")
    if os.path.exists(score_dir):
      pd.DataFrame.from_records([score]).to_csv(score_dir, header=False,  index=False, mode="a")
    else:
      pd.DataFrame.from_records([score]).to_csv(score_dir, index=False)

    pd.DataFrame.from_dict(results).to_csv(result_dir, index=False)

if __name__ == "__main__":
    main()

