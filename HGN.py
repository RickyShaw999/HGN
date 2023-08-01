import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


# 定义一个双向LSTM的类
class BiLSTM(nn.Module):  # 继承PyTorch的nn.Module类
    """
    BiLSTM类，实现了双向长短期记忆网络(Bi-directional Long Short Term Memory Networks)。

    Attributes:
        forward_lstm (nn.LSTM): 用于前向传播的LSTM层。
        backward_lstm (nn.LSTM): 用于反向传播的LSTM层。

    Args:
        hidden_size (int): LSTM隐藏层的尺寸。
    """

    def __init__(self, hidden_size: int):  # todo(hidden_size超参)
        super(BiLSTM, self).__init__()  # 调用父类的构造函数
        # 创建一个前向的LSTM，输入和输出的维度都是hidden_size的一半
        self.forward_lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        # 创建一个反向的LSTM，输入和输出的维度都是hidden_size的一半
        self.backward_lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor):
        """
        定义BiLSTM的前向计算方法。

        Args:
            x (torch.Tensor): 输入到BiLSTM的张量, [batch_size, sequence_length, feature_dim]。

        Returns:
            output (torch.Tensor): 经过BiLSTM处理的张量。
            (h1, c1), (h2, c2) (Tuple[Tensor, Tensor]): 前向和反向LSTM的隐藏状态和细胞状态。
        """
        # 获取输入张量的尺寸
        batch_size, max_len, feat_dim = x.shape
        # 前向传播
        out1, (h1, c1) = self.forward_lstm(x)
        # 初始化一个与输入同尺寸的零张量，存储反向传播的输入
        reverse_x = torch.zeros(
            [batch_size, max_len, feat_dim], dtype=torch.float32, device="cuda"
        )
        # 反转输入序列
        for i in range(max_len):
            reverse_x[:, i, :] = x[:, max_len - 1 - i, :]
        # 反向传播
        out2, (h2, c2) = self.backward_lstm(reverse_x)
        # 将前向和反向的输出拼接在一起
        output = torch.cat((out1, out2), 2)
        return output, (1, 1)  # 返回拼接后的输出和假设的隐藏状态


# 定义一个命名实体识别模型的类
class HGNER(nn.Module):  # 继承PyTorch的nn.Module类
    """
    HGNER类，实现了一个基于BERT和BiLSTM的命名实体识别模型。

    Attributes:
        bert (AutoModel): 使用预训练的BERT模型。
        dropout (nn.Dropout): dropout层。
        num_labels (int): 输出的标签数量。
        use_bilstm (bool): 是否使用双向LSTM。
        use_multiple_window (bool): 是否使用多窗口策略。
        windows_list (List[int]): 多窗口的大小列表。
        connect_type (str): 连接类型，可选'dot-att'或'mlp-att'。
        d_model (int): 模型的隐藏层维度。
        bilstm_layers (nn.ModuleList): 双向LSTM层的列表。
        linear (nn.Linear): 线性变换层。

    Args:
        args (Namespace): 命令行参数对象。
        num_labels (int): 输出的标签数量。
        hidden_dropout_prob (float, optional): dropout概率，默认为0.1。
        windows_list (List[int], optional): 多窗口的大小列表， 默认为None。
    """

    def __init__(
        self, args, num_labels: int, hidden_dropout_prob: float = 0.1, windows_list=None
    ):
        super(HGNER, self).__init__()  # 调用父类的构造函数

        # 从预训练的BERT模型加载配置
        config = AutoConfig.from_pretrained(args.bert_model)  # todo(bert_model超参)
        # 加载预训练的BERT模型
        self.bert = AutoModel.from_pretrained(args.bert_model)

        # 创建一个Dropout层，dropout的概率为hidden_dropout_prob
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # 设置标签的数量
        self.num_labels = num_labels

        # 是否使用双向LSTM
        self.use_bilstm = args.use_bilstm  # todo(超参)

        # 是否使用多窗口策略
        self.use_multiple_window = args.use_multiple_window  # todo(超参)
        # 设置窗口大小列表
        self.windows_list = windows_list
        # 设置连接类型
        self.connect_type = args.connect_type  # todo(超参)
        # 设置模型隐藏层的维度
        self.d_model = args.d_model  # todo(超参)

        # 如果使用多窗口策略，并且窗口大小列表不为空
        if self.use_multiple_window and self.windows_list != None:
            # 如果使用双向LSTM
            if self.use_bilstm:
                # 创建一个包含多个BiLSTM的ModuleList
                self.bilstm_layers = nn.ModuleList(
                    [BiLSTM(self.d_model) for _ in self.windows_list]
                )
            else:
                # 创建一个包含多个单向LSTM的ModuleList
                self.bilstm_layers = nn.ModuleList(
                    [
                        nn.LSTM(
                            self.d_model,
                            self.d_model,
                            num_layers=1,
                            bidirectional=False,
                            batch_first=True,
                        )
                        for _ in self.windows_list
                    ]
                )

            # 根据连接类型设置线性变换层
            if self.connect_type == "dot-att":
                self.linear = nn.Linear(self.d_model, self.num_labels)
            elif self.connect_type == "mlp-att":
                self.linear = nn.Linear(self.d_model, self.num_labels)
                self.Q = nn.Linear(self.d_model * (len(windows_list) + 1), self.d_model)
        else:
            # 如果不使用多窗口策略，或者窗口大小列表为空，创建一个线性变换层
            self.linear = nn.Linear(self.d_model, self.num_labels)

    def windows_sequence(
        self, sequence_output: torch.Tensor, windows: int, lstm_layer: nn.Module
    ):
        """
        根据窗口大小和LSTM层处理序列输出。

        Args:
            sequence_output (torch.Tensor): 输入到LSTM的张量。
            windows (int): 窗口大小。
            lstm_layer (nn.Module): LSTM层。

        Returns:
            local_final (torch.Tensor): 处理后的张量。
        """
        # 获取输入张量的尺寸
        batch_size, max_len, feat_dim = sequence_output.shape
        # 初始化一个与输入同尺寸的零张量，存储处理后的结果
        local_final = torch.zeros(
            [batch_size, max_len, feat_dim], dtype=torch.float32, device="cuda"
        )
        # 对每个时间步进行处理
        for i in range(max_len):
            # 初始化一个空的索引列表
            index_list = []
            # 对每个窗口进行处理
            for u in range(1, windows // 2 + 1):
                # 如果可以向前看，添加前一个时间步的索引
                if i - u >= 0:
                    index_list.append(i - u)
                # 如果可以向后看，添加后一个时间步的索引
                if i + u <= max_len - 1:
                    index_list.append(i + u)
            # 添加当前时间步的索引
            index_list.append(i)
            # 对索引列表进行排序
            index_list.sort()
            # 根据索引列表获取局部序列
            temp = sequence_output[:, index_list, :]
            # 使用LSTM处理局部序列
            out, (h, b) = lstm_layer(temp)
            # 获取最后一个时间步的输出
            local_f = out[:, -1, :]
            # 将输出存储到结果张量中
            local_final[:, i, :] = local_f
        return local_final

    # 代码的前面部分与上一次相同，所以省略了

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        valid_ids: torch.Tensor = None,
        attention_mask_label: torch.Tensor = None,
    ):
        """
        模型的前向传播。

        Args:
            input_ids (torch.Tensor): 输入的tokens的id。
            token_type_ids (torch.Tensor, optional): tokens的类型id，对应于两个句子的情况，默认为None。
            attention_mask (torch.Tensor, optional): 用于遮蔽padding tokens的二进制mask，默认为None。
            labels (torch.Tensor, optional): 真实的标签，默认为None。
            valid_ids (torch.Tensor, optional): 用于标识是原始输入还是添加的tokens的id，默认为None。
            attention_mask_label (torch.Tensor, optional): 用于遮蔽padding tokens的标签mask，默认为None。

        Returns:
            logits (torch.Tensor): 输出的logits。
            loss (torch.Tensor): 如果提供了labels，还会返回损失。
        """
        # 通过BERT模型获得的sequence输出
        sequence_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=None,
        )[0]
        # 获取sequence输出的尺寸
        batch_size, max_len, feat_dim = sequence_output.shape
        # 初始化一个和sequence输出相同尺寸的零张量
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32, device="cuda"
        )

        # 处理每一个batch的sequence输出
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:  # 如果当前位置的valid_id为1
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]  # 赋值给valid_output
        # 应用dropout
        sequence_output = self.dropout(valid_output)

        # 是否使用多窗口
        if self.use_multiple_window:
            mutiple_windows = []

            # 对每个窗口进行操作
            for i, window in enumerate(self.windows_list):
                if self.use_bilstm:  # 如果使用双向LSTM
                    local_final = self.windows_sequence(
                        sequence_output, window, self.bilstm_layers[i]
                    )  # 获取双向LSTM的输出
                mutiple_windows.append(local_final)

            # 处理连接类型
            if self.connect_type == "dot-att":
                muti_local_features = torch.stack(mutiple_windows, dim=2)  # 对不同窗口的输出堆叠
                sequence_output = sequence_output.unsqueeze(dim=2)  # 在第2维增加一个维度
                d_k = sequence_output.size(-1)  # 获取最后一维的尺寸
                attn = torch.matmul(
                    sequence_output, muti_local_features.permute(0, 1, 3, 2)
                ) / math.sqrt(
                    d_k
                )  # 计算attention
                attn = torch.softmax(attn, dim=-1)  # 在最后一维进行softmax操作
                local_features = torch.matmul(
                    attn, muti_local_features
                ).squeeze()  # 根据attention得到局部特征
                sequence_output = sequence_output.squeeze()  # 移除增加的维度
                sequence_output = sequence_output + local_features  # 与局部特征相加
            elif self.connect_type == "mlp-att":
                mutiple_windows.append(sequence_output)  # 将sequence_output添加到窗口列表
                muti_features = torch.cat(mutiple_windows, dim=-1)  # 将所有窗口的输出拼接
                muti_local_features = torch.stack(mutiple_windows, dim=2)  # 对不同窗口的输出堆叠
                query = self.Q(muti_features)  # 对拼接后的输出进行线性变换
                d_k = query.size(-1)  # 获取最后一维的尺寸
                query = query.unsqueeze(dim=2)  # 在第2维增加一个维度
                attn = torch.matmul(
                    query, muti_local_features.permute(0, 1, 3, 2)
                ) / math.sqrt(
                    d_k
                )  # 计算attention
                attn = torch.softmax(attn, dim=-1)  # 在最后一维进行softmax操作
                sequence_output = torch.matmul(
                    attn, muti_local_features
                ).squeeze()  # 根据attention得到输出

        # 将sequence_output通过线性层得到logits
        logits = self.linear(sequence_output)

        # 如果提供了labels，计算loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 创建交叉熵损失函数，忽略id为0的tokens
            # 如果提供了attention_mask_label，只考虑有效的部分
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1  # 获取有效位置的mask
                active_logits = logits.view(-1, self.num_labels)[
                    active_loss
                ]  # 获取有效位置的logits
                active_labels = labels.view(-1)[active_loss]  # 获取有效位置的labels
                loss = loss_fct(active_logits, active_labels)  # 计算损失
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )  # 直接计算损失
            return loss
        else:  # 如果没有提供labels，只返回logits
            return logits
