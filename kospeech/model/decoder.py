import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from kospeech.model.encoder import BaseRNN
from kospeech.model.attention import LocationAwareAttention, MultiHeadAttention


class Speller(BaseRNN):
    """
    Converts higher level features (from listener) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): the number of classfication
        max_length (int): a maximum allowed length for the sequence to be processed
        hidden_dim (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        num_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_dim): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: decoder_outputs
        - **decoder_outputs**: list of tensors containing the outputs of the decoding function.
    """
    KEY_ATTENTION_SCORE = 'attention_score'
    KEY_SEQUENCE_SYMBOL = 'sequence_symbol'

    def __init__(self, num_classes, max_length, hidden_dim, sos_id, eos_id, attn_mechanism='dot',
                 num_heads=4, num_layers=2, rnn_type='lstm', dropout_p=0.3, device=None):
        super(Speller, self).__init__(hidden_dim, hidden_dim, num_layers, rnn_type, dropout_p, False, device)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.acoutsic_weight = 0.9  # acoustic model weight
        self.language_weight = 0.1  # language model weight
        self.attn_mechanism = attn_mechanism
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(hidden_dim << 1, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)

        if attn_mechanism == 'loc':
            self.attention = LocationAwareAttention(hidden_dim, smoothing=True)
        elif attn_mechanism == 'dot':
            self.attention = MultiHeadAttention(hidden_dim, num_heads)
        else:
            raise ValueError("Unsupported attention: %s".format(attn_mechanism))

    def forward_step(self, input_var, hidden, encoder_outputs, attn):
        batch_size, output_lengths = input_var.size(0), input_var.size(1)
        input_var.cuda()
        embedded = self.embedding(input_var).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)
        context, attn = self.attention(output, encoder_outputs, attn)

        output = self.fc1(context.view(-1, self.hidden_dim << 1)).view(batch_size, -1, self.hidden_dim)
        output = self.fc2(torch.tanh(output).contiguous().view(-1, self.hidden_dim))

        predicted_softmax = F.log_softmax(output, dim=1)
        step_output = predicted_softmax.view(batch_size, output_lengths, -1).squeeze(1)

        return step_output, hidden, attn

    def forward(self, inputs, encoder_outputs, teacher_forcing_ratio=0.90, language_model=None):
        hidden, attn = None, None
        decoder_outputs, metadata = list(), dict()

        if not self.training:
            metadata[Speller.KEY_ATTENTION_SCORE] = list()
            metadata[Speller.KEY_SEQUENCE_SYMBOL] = list()

        inputs, batch_size, max_length = self.validate_args(inputs, encoder_outputs, teacher_forcing_ratio, language_model)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        use_language_model = True if language_model is not None else False

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)

            # Call forward_step() at every timestep when attention mechanism is location-aware
            # Because location-aware attention requires previous attention (alignment).
            if self.attn_mechanism == 'loc':
                for di in range(inputs.size(1)):
                    input_var = inputs[:, di].unsqueeze(1)
                    step_output, hidden, attn = self.forward_step(input_var, hidden, encoder_outputs, attn)
                    decoder_outputs.append(step_output)

            else:
                step_outputs, hidden, attn = self.forward_step(inputs, hidden, encoder_outputs, attn)

                for di in range(step_outputs.size(1)):
                    step_output = step_outputs[:, di, :]
                    decoder_outputs.append(step_output)

        else:
            input_var = inputs[:, 0].unsqueeze(1)
            prev_tokens = input_var.clone() if use_language_model else None

            for di in range(max_length):
                step_output, hidden, attn = self.forward_step(input_var, hidden, encoder_outputs, attn)

                if not self.training:
                    metadata[Speller.KEY_ATTENTION_SCORE].append(attn)
                    metadata[Speller.KEY_SEQUENCE_SYMBOL].append(input_var)

                    if use_language_model:
                        lm_step_output = language_model.forward_step(prev_tokens, None)[0][:, -1, :].squeeze(1)
                        step_output = step_output * self.acoustic_weight + lm_step_output * self.language_weight
                        prev_tokens = torch.cat([prev_tokens, step_output.topk(1)[1]], dim=1)

                decoder_outputs.append(step_output)
                input_var = decoder_outputs[-1].topk(1)[1]

        return decoder_outputs, metadata

    def validate_args(self, inputs, encoder_outputs, teacher_forcing_ratio, language_model):
        """ Validate arguments """
        batch_size = encoder_outputs.size(0)

        if inputs is None:  # inference
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        if language_model is not None:
            language_model.eval()

        return inputs, batch_size, max_length
