from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import time

# import truecase
import re
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
import truecase
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup, PreTrainedTokenizer,
)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from seqeval.metrics import classification_report
from HGN import HGNER

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)




class InputExample(object):
    """
    一个简单的序列分类的训练/测试例子。

    Attributes:
        guid (str): 示例的唯一id。
        text_a (str): 第一个序列的未经标记化的文本。对于单序列任务，只需指定这个序列。
        text_b (str, optional): 第二个序列的未经标记化的文本。只有在序列对任务中必须指定。
        label (str, optional): 示例的标签。对于训练和开发样本应指定，但对于测试样本不应指定。
        domain_label (str, optional): 域标签。
    """

    def __init__(
        self,
        guid: str,
        text_a: str,
        text_b: str = None,
        label: str = None,
        domain_label: str = None,
    ):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain_label = domain_label


class InputFeatures(object):
    """
    数据的一组特征。

    Attributes:
        input_ids (list): 输入的id列表。
        input_mask (list): 输入的掩码列表。
        segment_ids (list): 段落id列表。
        label_id (list): 标签id列表。
        valid_ids (list, optional): 有效id列表。
        label_mask (list, optional): 标签掩码列表。
        domain_label (list, optional): 域标签。
        seq_len (int, optional): 序列长度。
    """

    def __init__(
        self,
        input_ids: list,
        input_mask: list,
        segment_ids: list,
        label_id: list,
        valid_ids: list = None,
        label_mask: list = None,
        domain_label: list = None,
        seq_len: int = None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.domain_label = domain_label
        self.seq_len = seq_len


def readfile(filename: str, type_: str = None) -> list:
    """
    读取文件内容。

    Args:
        filename (str): 文件名。
        type_ (str, optional): 类型，预测时不为None，默认为None。

    Returns:
        data (list): 包含句子和标签的元组的列表。
    """
    f = open(filename)  # 打开文件
    data = []  # 初始化数据列表
    sentence = []  # 初始化句子列表
    label = []  # 初始化标签列表
    for line in f:  # 遍历每一行
        # 如果这一行长度为0，或者以'-DOCSTART'开始，或者第一个字符为"\n"
        if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == "\n":
            if len(sentence) > 0:  # 如果句子列表长度大于0
                assert len(sentence) == len(label)  # 确保句子和标签长度一致
                data.append((sentence, label))  # 添加到数据列表
                sentence = []  # 清空句子列表
                label = []  # 清空标签列表
            continue  # 跳过当前循环

        if type_ != "predict":  # 如果类型不是'predict'
            splits = line.split()  # 切分行
            sentence.append(splits[0])  # 添加到句子列表
            label.append(splits[-1])  # 添加到标签列表

        else:  # 如果类型是'predict'
            splits = line.strip().split()  # 去除行尾空格后切分行
            sentence.append(splits[0])  # 添加到句子列表
            label.append("O")  # 标签列表添加'O'

    if len(sentence) > 0:  # 如果句子列表长度大于0
        data.append((sentence, label))  # 添加到数据列表
        assert len(sentence) == len(label)  # 确保句子和标签长度一致
        sentence = []  # 清空句子列表
        label = []  # 清空标签列表
    return data  # 返回数据


def readfile_label(train_file, test_file, dev_file):
    """
    read file
    """
    label_dict = {}
    f_train = open(train_file)
    f_test = open(test_file)
    f_dev = open(dev_file)

    for line in f_train:
        temp = line.strip()
        if temp != "":

            splits = line.strip().split()
            if splits[-1] != "O":
                label_dict[splits[-1]] = 0

    for line in f_test:
        temp = line.strip()
        if temp != "":

            splits = line.strip().split()
            if splits[-1] != "O":
                label_dict[splits[-1]] = 0

    for line in f_dev:
        temp = line.strip()
        if temp != "":

            splits = line.strip().split()
            if splits[-1] != "O":
                label_dict[splits[-1]] = 0
    return label_dict


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_predict_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, label_list):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, type_=None):
        """Reads a tab separated value file."""
        return readfile(input_file, type_)


class NerProcessor(DataProcessor):
    """处理 CoNLL-2003 数据集的处理器。

    Args:
        DataProcessor: 数据处理器的基类。

    """

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """获取训练数据。

        Args:
            data_dir (str): 数据所在的文件夹路径。

        Returns:
            List[InputExample]: 返回 InputExample 类型的训练数据列表。

        """
        return self._create_examples(self._read_tsv(data_dir, type_="train"), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """获取开发（验证）数据。

        Args:
            data_dir (str): 数据所在的文件夹路径。

        Returns:
            List[InputExample]: 返回 InputExample 类型的开发（验证）数据列表。

        """
        return self._create_examples(self._read_tsv(data_dir, type_="dev"), "dev")

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        """获取测试数据。

        Args:
            data_dir (str): 数据所在的文件夹路径。

        Returns:
            List[InputExample]: 返回 InputExample 类型的测试数据列表。

        """
        return self._create_examples(self._read_tsv(data_dir, type_="test"), "test")

    def get_predict_examples(self, data_dir: str) -> List[InputExample]:
        """获取预测数据。

        Args:
            data_dir (str): 数据所在的文件夹路径。

        Returns:
            List[InputExample]: 返回 InputExample 类型的预测数据列表。

        """
        return self._create_examples(
            self._read_tsv(data_dir, type_="predict"), "predict"
        )

    def get_labels(self, label_list: List[str]) -> List[str]:
        """生成带有前缀的标签列表。

        Args:
            label_list (List[str]): 不带前缀的标签列表。

        Returns:
            List[str]: 带有 'B-', 'I-', 'E-', 'S-', '[CLS]', '[SEP]' 前缀的标签列表。

        """
        f_label_list = []
        f_label_list.append("O")
        for i in label_list:
            f_label_list.append("B-" + i)
            f_label_list.append("I-" + i)
            f_label_list.append("E-" + i)
            f_label_list.append("S-" + i)
        f_label_list.append("[CLS]")
        f_label_list.append("[SEP]")
        return f_label_list

    def _create_examples(self, lines: List[Tuple[str, str]], set_type: str) -> List[InputExample]:
        """根据输入的句子和标签创建样本。

        Args:
            lines (List[Tuple[str, str]]): 包含句子和对应标签的列表。
            set_type (str): 当前处理的数据集类型（'train'、'dev'、'test'、'predict'）。

        Returns:
            List[InputExample]: 返回创建的样本列表。

        """
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = "\t\t".join(sentence)
            text_b = None
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


def truecase_sentence(tokens: List[str]) -> List[str]:
    """将输入的词列表的每一个词转为正确的大小写形式。

    Args:
        tokens (List[str]): 输入的词列表。

    Returns:
        List[str]: 转化为正确的大小写形式后的词列表。

    """
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r"\b[A-Z\.\-]+\b", w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(" ".join(lst)).split()

        # the trucaser have its own tokenization ...
        # skip if the number of word dosen't match
        if len(parts) != len(word_lst):
            return tokens

        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens



def convert_examples_to_features(
    examples: List,
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer
) -> Tuple[List[InputFeatures], List[List[str]]]:
    """
    将数据文件加载到“InputBatch”列表中。

    Args:
        examples (List): 输入的样本，每个样本包含文本(text_a)和标签(label)，其中文本是以'\t\t'作为分隔符的。
        label_list (List[str]): 序列标注。
        max_seq_length (int): 序列的最大长度，超过该长度的序列将被截断。
        tokenizer (PreTrainedTokenizer): 预训练的tokenizer用于文本的编码。

    Returns:
        Tuple[List[InputFeatures], List[List[str]]]: 返回一个元组，包含转化后的特征和原始句子。
    """

    # 创建一个字典，将每个标签映射到一个唯一的数字，从1开始计数，注意这里也包含特殊标记[CLS]和[SEP]的编码
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []  # 用于保存处理后的样本特征
    ori_sents = []  # 用于保存原始句子

    for (ex_index, example) in enumerate(examples):
        # 以"\t\t"为分隔符拆分文本，获取所有单词
        textlist = example.text_a.split("\t\t")
        ori_sents.append(textlist)  # 保存原始句子

        # 获取当前样本的标签列表
        labellist = example.label

        tokens = []  # 保存分词后的所有token
        labels = []  # 对应每个token的标签
        valid = []  # 用于标记是否是有效的token
        label_mask = []  # 用于标记是否是有标签的token
        seq_len = []  # 保存每个句子的长度
        seq_len.append(len(textlist))

        for i, word in enumerate(textlist):
            # 对单词进行分词
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)  # 将第一个token的标签设为当前单词的标签
                    valid.append(1)  # 将第一个token的有效性设为1
                    label_mask.append(1)  # 将第一个token的标签掩码设为1
                else:
                    valid.append(0)  # 将其他token的有效性设为0

        # 如果token的数量超过最大序列长度-2，则对token进行截断，其中要留2个位置给特殊token
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0 : (max_seq_length - 2)]
            labels = labels[0 : (max_seq_length - 2)]
            valid = valid[0 : (max_seq_length - 2)]
            label_mask = label_mask[0 : (max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")  # 在序列开始添加CLS token
        segment_ids.append(0)  # CLS token的片段ID为0
        valid.insert(0, 1)  # CLS token是有效的
        label_mask.insert(0, 1)  # CLS token的标签掩码为1
        label_ids.append(label_map["[CLS]"])  # 添加CLS token的标签ID

        for i, token in enumerate(tokens):
            ntokens.append(token)  # 添加token
            segment_ids.append(0)  # 当前片段ID为0
            if len(labels) > i:
                try:
                    label_ids.append(label_map[labels[i]])  # 添加word的标签ID
                except:  # 用于调试
                    print(tokens)
                    print(labels)
                    time.sleep(100)

        ntokens.append("[SEP]")  # 在序列结束添加SEP token
        segment_ids.append(0)  # SEP token的片段ID为0
        valid.append(1)  # SEP token是有效的
        label_mask.append(1)  # SEP token的标签掩码为1
        label_ids.append(label_map["[SEP]"])  # 添加SEP token的标签ID

        # 将ntokens转化为模型可接受的input ids
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        # 创建attention mask
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        # 如果序列长度小于最大序列长度，用0进行填充
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        # 检查序列的长度是否符合预期
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # 将样本特征添加到features列表
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_ids,
                valid_ids=valid,
                label_mask=label_mask,
                seq_len=seq_len,
            )
        )

    return features, ori_sents

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = False
    torch.random.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_data_dir",  # todo(超参)
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--dev_data_dir",  # todo(超参)
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--test_data_dir",  # todo(超参)
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--gpu_id",  # todo(超参)
        default=None,
        nargs="+",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument("--use_crf", action="store_true", help="Whether use crf")  # todo(超参)

    parser.add_argument(
        "--bert_model",  # todo(超参)
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
    )
    parser.add_argument(
        "--task_name",  # todo(超参)
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--output_dir",  # todo(超参)
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--cache_dir",  # todo(超参)
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--label_list",  # todo(超参？)
        default=["O"],
        type=str,
        nargs="+",
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--max_seq_length",  # todo(超参)
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."  # todo(超参)
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval or not."  # todo(超参)
    )

    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run eval or not."  # todo(超参)
    )

    parser.add_argument(
        "--eval_on",  # todo(超参)
        default="dev",
        help="Whether to run eval on the dev set or test set.",
    )
    parser.add_argument(
        "--do_lower_case",  # todo(超参)
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--train_batch_size",  # todo(超参)
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Total batch size for eval."  # todo(超参)
    )
    parser.add_argument(
        "--learning_rate",  # todo(超参)
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",  # todo(超参)
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",  # todo(超参)
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight deay if we apply some."  # todo(超参)
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."  # todo(超参)
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."  # todo(超参)
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"  # todo(超参)
    )
    parser.add_argument(
        "--local_rank",  # todo(超参)
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"  # todo(超参)
    )
    parser.add_argument(
        "--gradient_accumulation_steps",  # todo(超参)
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",  # todo(超参)
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",  # todo(超参)
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--loss_scale",  # todo(超参)
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="Can be used for distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="Can be used for distant debugging."
    )

    parser.add_argument(
        "--hidden_dropout_prob", default=0.1, type=float, help="hidden_dropout_prob"  # todo(超参)
    )

    parser.add_argument("--window_size", default=-1, type=int, help="window_size")  # todo(超参)

    parser.add_argument(
        "--d_model", default=1024, type=int, help="pre-trained model size"
    )

    #####
    parser.add_argument(
        "--use_bilstm",  # todo(超参)
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--use_single_window",  # todo(超参)
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--use_multiple_window",  # todo(超参)
        action="store_true",
        help="Set this flag if you are using an multiple.",
    )

    parser.add_argument(
        "--use_global_lstm",  # todo(超参)
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--use_n_gram",  # todo(超参)
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument("--windows_list", type=str, default="", help="window list")  # todo(超参)
    parser.add_argument("--connect_type", type=str, default="add", help="window list")  # todo(超参)

    args = parser.parse_args()

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    # 首先检查是否在训练模式
    if args.do_train:
        # 如果输出目录不存在，则创建该目录
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # 创建日志处理器，用于将日志信息输出到文件
    handler = logging.FileHandler(args.output_dir + "/log.txt", encoding="UTF-8")
    # 将日志处理器添加到日志记录器
    logger.addHandler(handler)

    # 获取指定的GPU id，将其组合成一个字符串
    gpu_ids = ""
    for ids in args.gpu_id:
        gpu_ids = gpu_ids + str(ids) + ","

    # 设置环境变量，使得代码只在指定的GPU上运行
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # 如果用户指定了服务器IP和端口，那么就进行远程调试
    if args.server_ip and args.server_port:
        # 导入用于远程调试的模块
        import ptvsd

        print("Waiting for debugger attach")
        # 启动远程调试
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        # 等待调试器连接
        ptvsd.wait_for_attach()

    # 设置处理器，这里只设置了NER任务的处理器
    processors = {"ner": NerProcessor}

    # 检查是否使用了CUDA设备
    if args.local_rank == -1 or args.no_cuda:
        # 如果没有使用CUDA设备或者没有指定设备的话，那么就在可用的设备中选择一个进行训练
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        # 获取可用的GPU数量
        n_gpu = torch.cuda.device_count()
    else:
        # 如果用户指定了设备，那么就在指定的设备上进行训练
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # 初始化分布式训练环境
        torch.distributed.init_process_group(backend="nccl")

    # 记录一些关于设备、GPU数量、是否进行分布式训练以及是否使用了16位精度训练的信息
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    # 检查梯度积累步数的参数是否正确
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    # 计算真实的训练批次大小
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 设置随机种子，以保证实验的复现性
    setup_seed(args.seed)

    # 获取任务名称，并将其转化为小写形式
    task_name = args.task_name.lower()

    # 检查任务是否存在
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # 获取任务对应的处理器
    processor = processors[task_name]()

    # 从预训练的BERT模型中加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    # 初始化训练样本和优化步数
    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        # 如果处于训练模式，那么就获取训练样本，并计算优化步数
        train_examples = processor.get_train_examples(args.train_data_dir)
        num_train_optimization_steps = (
                int(
                    len(train_examples)
                    / args.train_batch_size
                    / args.gradient_accumulation_steps
                )
                * args.num_train_epochs
        )

        # 如果进行了分布式训练，那么就将优化步数除以进程数量
        if args.local_rank != -1:
            num_train_optimization_steps = (
                    num_train_optimization_steps // torch.distributed.get_world_size()
            )

    # 如果进行了分布式训练，那么就等待所有进程都达到这一点，再继续往下执行
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # 读取标签文件，并生成标签列表
    label_dict = readfile_label(
        args.train_data_dir, args.test_data_dir, args.dev_data_dir
    )
    label_list = []
    label_list.append("O")
    for keys, _ in label_dict.items():
        label_list.append(keys)
    label_list.append("[CLS]")
    label_list.append("[SEP]")

    # 计算标签的数量
    num_labels = len(label_list) + 1

    # 记录参数信息
    logger.info(args)

    # 初始化模型
    model = HGNER(
        args,
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        windows_list=[int(k) for k in args.windows_list.split("qq")]
        if args.windows_list
        else args.window_size,
    )

    # 计算模型的参数数量
    n_params = sum([p.nelement() for p in model.parameters()])
    print("n_params", n_params)

    # 如果进行了分布式训练，那么就等待所有进程都达到这一点，再继续往下执行
    if args.local_rank == 0:
        torch.distributed.barrier()

    # 将模型移动到指定的设备上
    model.to(device)


    # 将模型中的参数放入列表中，准备进行优化
    param_optimizer = list(model.named_parameters())

    # 不进行权重衰减的参数类型，这里包括偏置项和LayerNorm层的权重
    no_decay = ["bias", "LayerNorm.weight"]

    # 将需要进行权重衰减和不需要进行权重衰减的参数分开
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],  # 需要权重衰减的参数
            "weight_decay": args.weight_decay,  # 权重衰减的比例
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],  # 不需要权重衰减的参数
            "weight_decay": 0.0,  # 权重衰减比例为0
        },
    ]

    # 计算预热步数，即开始进行学习率衰减的步数
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)

    # 使用AdamW优化器，设置学习率和epsilon值
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    # 设置学习率的调度器，进行线性预热
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps,
    )

    # 如果使用半精度训练，则引入apex库
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        # 使用apex库的amp进行自动混合精度训练
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # 如果使用多个GPU进行训练，则使用DataParallel进行并行计算
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 如果使用分布式训练，则使用DistributedDataParallel进行并行计算
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # 初始化全局步数、训练步数和训练损失
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # 为标签建立一个映射，从1开始编号
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    # 在日志中记录标签映射信息
    logger.info("*** Label map ***")
    logger.info(label_map)
    logger.info("*******************************************")

    # 初始化最佳epoch和最佳P/R/F1值
    best_epoch = -1
    best_p = -1
    best_r = -1
    best_f = -1
    best_test_f = -1
    best_eval_f = -1

    # 如果进行训练，则加载训练数据
    if args.do_train:
        train_features, _ = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer
        )

        # 在日志中记录训练信息
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        # 准备训练数据的各项输入，包括输入ID、输入mask、分段ID、标签ID、有效ID、标签mask以及序列长度
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_seq_lens = torch.tensor([f.seq_len for f in train_features], dtype=torch.long)

        # 将训练数据打包成TensorDataset
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids, all_seq_lens)


        # load valid data

        eval_examples = processor.get_dev_examples(args.dev_data_dir)
        eval_features, _ = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
        )
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
        all_valid_ids = torch.tensor(
            [f.valid_ids for f in eval_features], dtype=torch.long
        )
        all_lmask_ids = torch.tensor(
            [f.label_mask for f in eval_features], dtype=torch.long
        )
        # all_domain_l = torch.tensor([f.domain_label for f in eval_features], dtype=torch.long)
        all_seq_lens = torch.tensor(
            [f.seq_len for f in eval_features], dtype=torch.long
        )
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_valid_ids,
            all_lmask_ids,
            all_seq_lens,
        )

        # load test data
        test_examples = processor.get_test_examples(args.test_data_dir)
        test_features, _ = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer
        )
        all_input_ids_dev = torch.tensor(
            [f.input_ids for f in test_features], dtype=torch.long
        )
        all_input_mask_dev = torch.tensor(
            [f.input_mask for f in test_features], dtype=torch.long
        )
        all_segment_ids_dev = torch.tensor(
            [f.segment_ids for f in test_features], dtype=torch.long
        )
        all_label_ids_dev = torch.tensor(
            [f.label_id for f in test_features], dtype=torch.long
        )
        all_valid_ids_dev = torch.tensor(
            [f.valid_ids for f in test_features], dtype=torch.long
        )
        all_lmask_ids_dev = torch.tensor(
            [f.label_mask for f in test_features], dtype=torch.long
        )
        # all_domain_l = torch.tensor([f.domain_label for f in test_features], dtype=torch.long)
        all_seq_lens_dev = torch.tensor(
            [f.seq_len for f in test_features], dtype=torch.long
        )
        test_data = TensorDataset(
            all_input_ids_dev,
            all_input_mask_dev,
            all_segment_ids_dev,
            all_label_ids_dev,
            all_valid_ids_dev,
            all_lmask_ids_dev,
            all_seq_lens_dev,
        )

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.eval_batch_size
        )

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

        test_f1 = []
        dev_f1 = []

        for epoch_ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                # begin_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    label_ids,
                    valid_ids,
                    l_mask,
                    seq_len,
                ) = batch
                loss = model(
                    input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask
                )  # , seq_len=seq_len)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                # end_time = time.time()
                # print('one step时间',end_time-begin_time)
            # eval in each epoch.
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                valid_ids,
                l_mask,
                seq_len,
            ) in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                seq_len = seq_len.to(device)
                # domain_l = domain_l.to(device)

                with torch.no_grad():
                    logits = model(
                        input_ids,
                        segment_ids,
                        input_mask,
                        valid_ids=valid_ids,
                        attention_mask_label=l_mask,
                    )  # , seq_len=seq_len)

                if not args.use_crf:
                    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to("cpu").numpy()
                input_mask = input_mask.to("cpu").numpy()

                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_map):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:

                            temp_1.append(label_map[label_ids[i][j]])
                            try:
                                temp_2.append(label_map[logits[i][j]])
                            except:
                                temp_2.append("O")
                            # temp_2.append(label_map[logits[i][j]])

            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n******evaluate on the dev data*******")
            logger.info("\n%s", report)
            temp = report.split("\n")[-3]
            f_eval = eval(temp.split()[-2])
            dev_f1.append(f_eval)

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

            # if os.path.exists(output_eval_file):
            with open(output_eval_file, "a") as writer:
                # logger.info("***** Eval results *****")
                # logger.info("=======token level========")
                # logger.info("\n%s", report)
                # logger.info("=======token level========")
                writer.write("*******************epoch*******" + str(epoch_) + "\n")
                writer.write(report + "\n")

            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            for (
                input_ids,
                input_mask,
                segment_ids,
                label_ids,
                valid_ids,
                l_mask,
                seq_len,
            ) in test_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                seq_len = seq_len
                # domain_l = domain_l.to(device)

                with torch.no_grad():
                    logits = model(
                        input_ids,
                        segment_ids,
                        input_mask,
                        valid_ids=valid_ids,
                        attention_mask_label=l_mask,
                    )
                    shape = logits.shape
                    if len(shape) < 3:
                        logits = logits.unsqueeze(dim=0)

                try:
                    if not args.use_crf:
                        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to("cpu").numpy()
                    input_mask = input_mask.to("cpu").numpy()
                except:
                    import pdb

                    pdb.set_trace()

                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_map):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            # print(temp_2)
                            # time.sleep(5)
                            break
                        else:
                            temp_1.append(label_map[label_ids[i][j]])
                            try:
                                temp_2.append(label_map[logits[i][j]])
                            except:
                                temp_2.append("O")
                            # temp_2.append(label_map[logits[i][j]])

            report = classification_report(y_true, y_pred, digits=4)

            logger.info("\n******evaluate on the test data*******")
            logger.info("\n%s", report)
            temp = report.split("\n")[-3]
            f_test = eval(temp.split()[-2])
            test_f1.append(f_test)

            output_eval_file_t = os.path.join(args.output_dir, "test_results.txt")

            # if os.path.exists(output_eval_file):
            with open(output_eval_file_t, "a") as writer2:
                # logger.info("***** Eval results *****")
                # logger.info("=======token level========")
                # logger.info("\n%s", report)
                # logger.info("=======token level========")
                writer2.write("*******************epoch*******" + str(epoch_) + "\n")
                writer2.write(report + "\n")

        # Load a trained model and config that you have fine-tuned
        output_f1_test = os.path.join(args.output_dir, "f1_score_epoch.txt")
        with open(output_f1_test, "w") as writer1:
            for i, j in zip(test_f1, dev_f1):
                writer1.write(str(i) + "\t" + str(j) + "\n")
            writer1.write("\n")
            writer1.write(str(best_test_f))

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        model = Ner.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        model.to(device)
        # if args.eval_on == "dev":
        #    eval_examples = processor.get_dev_examples(args.data_dir)
        # elif args.eval_on == "test":
        eval_examples = processor.get_test_examples(args.test_data_dir)
        # else:
        #    raise ValueError("eval on dev or test set only")
        eval_features, _ = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
        all_valid_ids = torch.tensor(
            [f.valid_ids for f in eval_features], dtype=torch.long
        )
        all_lmask_ids = torch.tensor(
            [f.label_mask for f in eval_features], dtype=torch.long
        )
        eval_data = TensorDataset(
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            all_label_ids,
            all_valid_ids,
            all_lmask_ids,
        )
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
        )
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(
            eval_dataloader, desc="Evaluating"
        ):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(
                    input_ids,
                    segment_ids,
                    input_mask,
                    valid_ids=valid_ids,
                    attention_mask_label=l_mask,
                )

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to("cpu").numpy()
            input_mask = input_mask.to("cpu").numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)

        output_result_file = os.path.join(args.output_dir, "text_results.txt")
        with open(output_result_file, "w") as writer:
            for i, j in zip(y_true, y_pred):
                for n, m in zip(i, j):
                    writer.write(n + "\t" + m + "\n")
                writer.write("\n")

    if args.do_predict:
        model_best = torch.load(args.output_dir + "/model.pt")
        tokenizer = AutoTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
        model_best.to(device)

        #

        test_examples = processor.get_test_examples(args.test_data_dir)
        test_features, ori_sents = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer
        )
        all_input_ids_dev = torch.tensor(
            [f.input_ids for f in test_features], dtype=torch.long
        )
        all_input_mask_dev = torch.tensor(
            [f.input_mask for f in test_features], dtype=torch.long
        )
        all_segment_ids_dev = torch.tensor(
            [f.segment_ids for f in test_features], dtype=torch.long
        )
        all_label_ids_dev = torch.tensor(
            [f.label_id for f in test_features], dtype=torch.long
        )
        all_valid_ids_dev = torch.tensor(
            [f.valid_ids for f in test_features], dtype=torch.long
        )
        all_lmask_ids_dev = torch.tensor(
            [f.label_mask for f in test_features], dtype=torch.long
        )
        # all_domain_l = torch.tensor([f.domain_label for f in test_features], dtype=torch.long)
        all_seq_lens_dev = torch.tensor(
            [f.seq_len for f in test_features], dtype=torch.long
        )
        test_data = TensorDataset(
            all_input_ids_dev,
            all_input_mask_dev,
            all_segment_ids_dev,
            all_label_ids_dev,
            all_valid_ids_dev,
            all_lmask_ids_dev,
            all_seq_lens_dev,
        )

        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=args.eval_batch_size
        )
        model_best.eval()

        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for (
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            valid_ids,
            l_mask,
            seq_len,
        ) in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)
            seq_len = seq_len
            # domain_l = domain_l.to(device)

            with torch.no_grad():
                logits = model_best(
                    input_ids,
                    segment_ids,
                    input_mask,
                    valid_ids=valid_ids,
                    attention_mask_label=l_mask,
                )
                shape = logits.shape
                if len(shape) < 3:
                    logits = logits.unsqueeze(dim=0)

            if not args.use_crf:
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to("cpu").numpy()
            input_mask = input_mask.to("cpu").numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        # print(temp_2)
                        # time.sleep(5)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        try:
                            temp_2.append(label_map[logits[i][j]])
                        except:
                            temp_2.append("O")
                        # temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred, digits=4)

        logger.info("\n******evaluate on the test data*******")
        logger.info("\n%s", report)

        # for i, sents in enumerate(ori_sents):
        #    temp = []
        #    temp2 = []
        #    for j,m in enumerate(sents):
        #       try:
        #         temp.append(label_map[logits[i][j]])
        #       except:
        #         temp.append('O')
        #       temp2.append(label_map[label_ids[i][j]])
        # y_pred.append(temp)
        # y_true.append(temp2)

        output_result_file = os.path.join(args.output_dir, "predict_results.txt")
        with open(output_result_file, "w") as writer:
            for i, j, k in zip(ori_sents, y_true, y_pred):
                for n, m, l in zip(i, j, k):
                    writer.write(n + " " + m + " " + l + "\n")
                writer.write("\n")


if __name__ == "__main__":
    main()
