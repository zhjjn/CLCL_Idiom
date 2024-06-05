import warnings
import csv
import pandas as pd
import os
from datetime import datetime
import logging
import random
import glob
import string
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationArgs
from classification_model.classification_model_cts import ClassificationModel
import torch.multiprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

MODEL_CLASSES = {
    "offline": ClassificationModel,
    "vanilla": ClassificationModel,
}

def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string

def remove_punctuation(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.translate(str.maketrans('', '', string.punctuation))
    )
    return out_string

def transfer_label(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = 0
    out_label = 0 if out_string == "i" else 1
    return out_label

parser = argparse.ArgumentParser()
parser.add_argument('--DataStream', action='store', dest='data_stream',help='Specify the data stream')
parser.add_argument('--DataVersion', action='store', dest='data_version',help='Specify the data version')
parser.add_argument('--Model', action='store', dest='model',help='Specify the model used')
parser.add_argument('--Mode', action='store', dest='mode',help='Specify the mode used')
parser.add_argument('--DataType', action='store', dest='data_type',help='Specify the data type')
parser.add_argument('--Round', action='store', dest='round',help='Specify the round')
parser.add_argument('--Order', action='store', dest='order',help='Specify the order')
parser.add_argument('--Sample', action='store', dest='sample',help='Specify the sampling method')
parser.add_argument('--CTS', action='store', dest='cts',help='Specify if contrastive learning is used')

args = parser.parse_args()

data_file = "./data/MAGPIE_processed_data_cts_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_Round" + args.round + ".csv"

if args.mode == 'dist_eval':
    
    model_args = ClassificationArgs()
    model_args.do_sample = True
    model_args.eval_batch_size = 16
    model_args.train_batch_size = 16
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_length = 256
    model_args.max_seq_length = 256
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
else:
    train_data = pd.read_csv(data_file).groupby("Split").get_group('training')
    valid_data = pd.read_csv(data_file).groupby("Split").get_group('development')
    test_data = pd.read_csv(data_file).groupby("Split").get_group('test')

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)

    train_df = train_data[["Idiom", "Context_3_Original", 'Idiom_Pos', 'Context_Pos', 'Idiom_Neg', 'Context_Neg', "Label"]]
    eval_df = valid_data[["Idiom", "Context_3_Original", "Label"]]
    test_df = test_data[["Idiom", "Context_3_Original", "Label"]]

    train_df = train_df.rename(
        columns={"Context_3_Original": "text_a", "Idiom": "text_b", "Context_Pos": "text_a_pos", "Idiom_Pos": "text_b_pos", "Context_Neg": "text_a_neg", "Idiom_Neg": "text_b_neg", "Label": "labels"}
    )

    eval_df = eval_df.rename(
        columns={"Context_3_Original": "text_a", "Idiom": "text_b", "Label": "labels"}
    )

    test_df = test_df.rename(
        columns={"Context_3_Original": "text_a", "Idiom": "text_b", "Label": "labels"}
    )
    
    
    train_df['text_a'] = train_df["text_a"].apply(remove_punctuation)
    train_df['text_b'] = train_df["text_b"].apply(remove_punctuation)
    train_df['text_a_pos'] = train_df["text_a_pos"].apply(remove_punctuation)
    train_df['text_b_pos'] = train_df["text_b_pos"].apply(remove_punctuation)
    train_df['text_a_neg'] = train_df["text_a_neg"].apply(remove_punctuation)
    train_df['text_b_neg'] = train_df["text_b_neg"].apply(remove_punctuation)
    
    eval_df['text_a'] = eval_df["text_a"].apply(remove_punctuation)
    eval_df['text_b'] = eval_df["text_b"].apply(remove_punctuation)
    
    test_df['text_a'] = test_df["text_a"].apply(remove_punctuation)
    test_df['text_b'] = test_df["text_b"].apply(remove_punctuation)

    #train_df["text_a"] = train_df["text_a"].apply(clean_unnecessary_spaces)
    train_df["labels"] = train_df["labels"].apply(transfer_label)

    #eval_df["text_a"] = eval_df["text_a"].apply(clean_unnecessary_spaces)
    eval_df["labels"] = eval_df["labels"].apply(transfer_label)

    test_df["labels"] = test_df["labels"].apply(transfer_label)
    #test_df = vmwe_test_df

    if args.model == "offline":
        train_df = train_df.sample(frac=1).reset_index(drop=True)

    model_args = ClassificationArgs()
    model_args.do_sample = True
    model_args.eval_batch_size = 32
    model_args.train_batch_size = 16
    #model_args.save_steps = int(len(train_df.values) / (model_args.train_batch_size * 10))
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = int(len(train_df.values) / (model_args.train_batch_size * 10))
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 1e-5
    model_args.max_length = 256
    model_args.max_seq_length = 256
    if args.model == "offline":
        model_args.num_train_epochs = 20
    else:
        model_args.num_train_epochs = 20
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = True
    model_args.save_steps = -1
    model_args.use_multiprocessing = False
    model_args.early_stopping_metric = 'mcc'
    model_args.early_stopping_metric_minimize = False
    model_args.n_gpu = 2
    
model_args.output_dir = "outputs_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_"+ args.model + "_" + args.order + "_Round" + args.round + "_" + args.sample + "_CTS_" + args.cts
model_args.best_model_dir = model_args.output_dir + "/best_model"

base_dir = "outputs_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_"+ args.model + "_" + args.order + "_Round" + str(int(args.round)-1) + "_" + args.sample + "_CTS_" + args.cts
base_best_dir = base_dir + "/best_model"


print("=================================" + args.data_stream + " " + args.data_type + " " + args.data_version + "=================================")

# Train the model
if args.mode == 'dist_eval':
    
    if args.round != '0':
        #ckpt = glob.glob(base_dir + "/checkpoint*epoch-1*")[0]
        model = ClassificationModel(
                model_type ="roberta",
                model_name= model_args.best_model_dir,
                args=model_args,
                )
    else:
        model = ClassificationModel(
                model_type ="roberta",
                model_name= 'roberta-base',
                args=model_args,
                )
        
    data_all = pd.read_csv(data_file)
    
    idiom_list = list(set(data_all['Idiom'].tolist()))
    diff_list = []
    fig_list = []

    for idiom in idiom_list:
        data = data_all.groupby("Idiom").get_group(idiom)
        test_df = data[["Idiom", "Context_3_Original", "Label"]]

        test_df = test_df.rename(
            columns={"Context_3_Original": "text_a", "Idiom": "text_b", "Label": "labels"}
        )

        test_df["labels"] = test_df["labels"].apply(transfer_label)
        diffs = model.eval_model_difficulty(test_df)
        diffs = diffs.item()
        #if str(diffs) == 'nan':
        #    diffs = 0
        print(idiom, diffs)
        if 1 in test_df["labels"].tolist():
            diff_list.append((idiom, diffs))
        else:
            fig_list.append((idiom, diffs))
    
    if args.order == 'Acending':
        sorted_diff_list = sorted(diff_list, key = lambda x: x[1])
        sorted_fig_list = sorted(fig_list, key = lambda x: x[1])
    else:
        sorted_diff_list = sorted(diff_list, key = lambda x: x[1], reverse = True)
        sorted_fig_list = sorted(fig_list, key = lambda x: x[1], reverse = True)
    
    for fig_tuple in sorted_fig_list:
      
        # choosing index to enter element
        index = random.randint(0, len(sorted_diff_list))

        # reforming list and getting random element to add
        sorted_diff_list = sorted_diff_list[:index] + [fig_tuple] + sorted_diff_list[index:]
 
    headers = list(data_all.columns)
    #if 'Perplexity' not in headers:
    #    headers.append('Perplexity')
    #if 'Dist' not in headers:
    #    headers.append('Dist')
    output = []
    
    for idiom, _ in sorted_diff_list:
        data = data_all.groupby("Idiom").get_group(idiom)
        data = data.sample(frac=1).reset_index(drop=True)
        for d in data.values:
            output.append(list(d))
    
    with open("./data/MAGPIE_processed_data_cts_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_Round" + str(int(args.round) + 1) + ".csv",'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(output)
        
        
elif args.mode == "train":
    if args.round == "1" or args.round == '0':
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= "roberta-base",
            cts = args.cts,
            cl = args.model,
            args=model_args,
        )
    else:
        #ckpt = glob.glob(base_dir + "/checkpoint*epoch-1*")[0]
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name=base_best_dir,
            cts = args.cts,
            cl = args.model,
            args=model_args,
        )
        
        
    idiom_list = []
    for idm in train_df['text_b'].tolist():
        if idm not in idiom_list:
            idiom_list.append(idm)
    diff_list = []
    fig_list = []
    
    if args.sample == 'downsample':
        for idiom in idiom_list:
            data = train_df.groupby("text_b").get_group(idiom)
            #if str(diffs) == 'nan':
            #    diffs = 0
            if 1 in data["labels"].tolist():
                lit_data = data.groupby("labels").get_group(1)
                try:
                    fig_data = list(data.groupby("labels").get_group(0).values)
                    sampled = random.sample(fig_data, len(fig_data) // 3 + 1)
                    comb_data = sampled + list(lit_data.values)
                except:
                    comb_data = list(lit_data.values)
                random.shuffle(comb_data)
                diff_list += comb_data
            else:
                cand = list(data.values)
                sampled = random.sample(cand, len(cand) // 3)
                fig_list += sampled
    elif args.sample == 'oversample':
        for idiom in idiom_list:
            data = train_df.groupby("text_b").get_group(idiom)
            #if str(diffs) == 'nan':
            #    diffs = 0
            if 1 in data["labels"].tolist():
                lit_data = data.groupby("labels").get_group(1)
                try:
                    fig_data = list(data.groupby("labels").get_group(0).values)
                    comb_data = fig_data + list(lit_data.values) + list(lit_data.values) + list(lit_data.values)
                except:
                    comb_data = list(lit_data.values) + list(lit_data.values) + list(lit_data.values)
                #random.shuffle(comb_data)
                diff_list += comb_data
            else:
                cand = list(data.values)
                fig_list += cand

    if args.sample == 'oversample' or args.sample == 'downsample':
        for d in fig_list:

            # choosing index to enter element
            index = random.randint(0, len(diff_list))

            # reforming list and getting random element to add
            diff_list = diff_list[:index] + [d] + diff_list[index:]

        oversampled_train_df = pd.DataFrame(diff_list, columns=['text_b', 'text_a', 'text_b_pos', 'text_a_pos', 'text_b_neg', 'text_a_neg', 'labels'])
    elif args.sample == 'nosample':
        oversampled_train_df = train_df
    print(oversampled_train_df[:64])
    print(oversampled_train_df['labels'].value_counts())
    
    model.train_model(oversampled_train_df, eval_df = eval_df)
elif args.mode == "inference":
    
    #ckpt = glob.glob(model_args.output_dir + "/checkpoint*epoch-1*")[0]

    if args.model == "offline":

        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= model_args.output_dir + "/best_model",
            cts = args.cts,
            cl = args.model,
            args=model_args,
            )


        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print("===========Offline Multipass Results==========")
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()
        
        print(classification_report(labels, preds, labels = [0,1], digits=4))

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_cts_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_multipass" + "_" + args.sample + ".csv",'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)
    else:
        print(model_args.output_dir)
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= model_args.output_dir + "/best_model",
            cts = args.cts,
            cl = args.model,
            args=model_args,
            )


        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()
        
        print(classification_report(labels, preds, labels = [0,1], digits=4))

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_cts_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_Round" + args.round + "_" + args.order + "_" + args.sample + "_CTS_" + args.cts + ".csv",'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)
            
"""        
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= model_args.output_dir + "/best_model",
            args=model_args,
            )


        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print("===========CL Multipass Results==========")
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_VMWE_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_multipass.csv",'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)
else:
    model.train_model(train_df, eval_df = eval_df)
    ckpt = glob.glob(model_args.output_dir + "/checkpoint*epoch-1*")[0]

    if args.model == "offline":
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= ckpt,
            args=model_args,
            )

        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print("===========Offline Onepass Results==========")
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_onepass.csv",'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)

        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= model_args.output_dir + "/best_model",
            args=model_args,
            )


        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print("===========Offline Multipass Results==========")
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_multipass.csv",'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)
    else:
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= ckpt,
            args=model_args,
            )


        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print("===========CL Onepass Results==========")
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_onepass.csv",'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)
            
        
        model = MODEL_CLASSES[args.model](
            model_type="roberta",
            model_name= model_args.output_dir + "/best_model",
            args=model_args,
            )


        result, model_outputs, wrong_predictions = model.eval_model(test_df)
        print("===========CL Multipass Results==========")
        print(result)

        output_headers = ['Idiom', 'Sentence', 'Label', 'Preds']

        preds = [int(i[0] < i[1]) for i in model_outputs.tolist()]

        idioms = test_df["text_b"].tolist()
        sentences = test_df["text_a"].tolist()
        labels = test_df["labels"].tolist()

        output_row = []
        for i, s, l, p in zip(idioms, sentences, labels, preds):
            output_row.append([i, s, l, p])

        with open("./results/Output_" + args.data_type + "_" + args.data_stream + "_" + args.data_version + "_" + args.model + "_multipass.csv",'w')as f:
            f_csv = csv.writer(f)
            f_csv.writerow(output_headers)
            f_csv.writerows(output_row)
            
"""