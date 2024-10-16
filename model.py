# -*- coding: utf-8 -*-


import os
import json
import re
import npdl
import numpy as np
from npdl.initializations import _one

  # Carga de diccionarios token2idx e idx2token desde archivos JSON
token2idx_path = "data/token2idx.json"
idx2token_path = "data/idx2token.json"
param_path = "data/params.npy"
max_sent_size = np.int32(50)
idx_start = np.int32(1)
idx_end = np.int32(2)
idx_unk = np.int32(3)  # unknown

token_start = "<start>"
token_end = "<end>"
token_unk = "<unk>"

if not os.path.exists(token2idx_path) or \
        not os.path.exists(idx2token_path) or \
        not os.path.exists(param_path):
    raise ValueError('Please download pre-trained models and put them in "data" directory.')


class Utils:
    token2idx = json.load(open(token2idx_path))
    idx2token = json.load(open(idx2token_path))

    @staticmethod
    def clearn_str(s):
        """
        Limpia la cadena de texto eliminando caracteres no deseados y espacios adicionales.

        Args:
            s (str): Cadena de texto a limpiar.

        Returns:
            str: Cadena de texto limpia.
        """
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\-", "", s)
        return s

    @staticmethod
    def load_params(hi):
        """
        Carga parámetros del modelo desde archivos numpy.

        Args:
            hi (int): Dimensión de los estados ocultos.

        Returns:
            dict: Diccionario de parámetros del modelo.
        """
        embed_words, \
        linear_weights, \
        en_lstm1_W, en_lstm1_U, en_lstm1_b, \
        en_lstm2_W, en_lstm2_U, en_lstm2_b, \
        de_lstm1_W, de_lstm1_U, de_lstm1_b, \
        de_lstm2_W, de_lstm2_U, de_lstm2_b = np.load(param_path, allow_pickle=True)

        params = {'embed_words': embed_words,

                  'en_lstm1': {'U_f': en_lstm1_W[:, : hi * 1],
                               'U_i': en_lstm1_W[:, hi * 1: hi * 2],
                               'U_o': en_lstm1_W[:, hi * 2: hi * 3],
                               'U_g': en_lstm1_W[:, hi * 3:],

                               'W_f': en_lstm1_U[:, : hi * 1],
                               'W_i': en_lstm1_U[:, hi * 1: hi * 2],
                               'W_o': en_lstm1_U[:, hi * 2: hi * 3],
                               'W_g': en_lstm1_U[:, hi * 3:],

                               'b_f': en_lstm1_b[: hi * 1],
                               'b_i': en_lstm1_b[hi * 1: hi * 2],
                               'b_o': en_lstm1_b[hi * 2: hi * 3],
                               'b_g': en_lstm1_b[hi * 3:]},

                  'en_lstm2': {'U_f': en_lstm2_W[:, : hi * 1],
                               'U_i': en_lstm2_W[:, hi * 1: hi * 2],
                               'U_o': en_lstm2_W[:, hi * 2: hi * 3],
                               'U_g': en_lstm2_W[:, hi * 3:],

                               'W_f': en_lstm2_U[:, : hi * 1],
                               'W_i': en_lstm2_U[:, hi * 1: hi * 2],
                               'W_o': en_lstm2_U[:, hi * 2: hi * 3],
                               'W_g': en_lstm2_U[:, hi * 3:],

                               'b_f': en_lstm2_b[: hi * 1],
                               'b_i': en_lstm2_b[hi * 1: hi * 2],
                               'b_o': en_lstm2_b[hi * 2: hi * 3],
                               'b_g': en_lstm2_b[hi * 3:]},

                  'de_lstm1': {'U_f': de_lstm1_W[:, : hi * 1],
                               'U_i': de_lstm1_W[:, hi * 1: hi * 2],
                               'U_o': de_lstm1_W[:, hi * 2: hi * 3],
                               'U_g': de_lstm1_W[:, hi * 3:],

                               'W_f': de_lstm1_U[:, : hi * 1],
                               'W_i': de_lstm1_U[:, hi * 1: hi * 2],
                               'W_o': de_lstm1_U[:, hi * 2: hi * 3],
                               'W_g': de_lstm1_U[:, hi * 3:],

                               'b_f': de_lstm1_b[: hi * 1],
                               'b_i': de_lstm1_b[hi * 1: hi * 2],
                               'b_o': de_lstm1_b[hi * 2: hi * 3],
                               'b_g': de_lstm1_b[hi * 3:]},

                  'de_lstm2': {'U_f': de_lstm2_W[:, : hi * 1],
                               'U_i': de_lstm2_W[:, hi * 1: hi * 2],
                               'U_o': de_lstm2_W[:, hi * 2: hi * 3],
                               'U_g': de_lstm2_W[:, hi * 3:],

                               'W_f': de_lstm2_U[:, : hi * 1],
                               'W_i': de_lstm2_U[:, hi * 1: hi * 2],
                               'W_o': de_lstm2_U[:, hi * 2: hi * 3],
                               'W_g': de_lstm2_U[:, hi * 3:],

                               'b_f': de_lstm2_b[: hi * 1],
                               'b_i': de_lstm2_b[hi * 1: hi * 2],
                               'b_o': de_lstm2_b[hi * 2: hi * 3],
                               'b_g': de_lstm2_b[hi * 3:]},

                  'linear': linear_weights}

        return params

    @staticmethod
    def tokenize(s):
        """
        Tokeniza una cadena de texto.

        Args:
            s (str): Cadena de texto a tokenizar.

        Returns:
            list: Lista de tokens.
        """
        s = Utils.clearn_str(s)  # Limpia la cadena de texto
        return s.strip().split(" ") # Divide la cadena por espacios

    # Métodos tokens2idxs, idxs2tokens, cut_and_pad y cut_end son similares
    # en su funcionamiento y toman listas de tokens o índices como entrada
    @staticmethod
    def tokens2idxs(tokens):
        rev = [str(Utils.token2idx.get(t, 3)) for t in tokens]  # default to <unk>
        return rev

    @staticmethod
    def idxs2tokens(idxs):
        rez = []
        for idx in idxs:
            if str(idx) in Utils.idx2token:
                rez.append(Utils.idx2token[str(idx)])
        return rez

    @staticmethod
    def cut_and_pad(ilist, max_size=max_sent_size):
        ilist = ilist[:max_size]
        rez = ilist + ["0"] * (max_size - len(ilist))
        return rez

    @staticmethod
    def cut_end(s):
        fid = s.find(token_end)
        if fid == -1:
            return s
        elif fid == 0:
            return ""
        else:
            return s[:fid - 1]

    @staticmethod
    def get_mask(data):
        """
        Genera una máscara indicando la presencia de datos.

        Args:
            data (numpy.ndarray): Datos de entrada.

        Returns:
            numpy.ndarray: Máscara indicando presencia de datos.
        """
        mask = (np.not_equal(data, 0)).astype("int32")
        return mask


class Seq2Seq:
    def __init__(self, hidden_size=512, nb_seq=max_sent_size):
        """
        Inicializa la arquitectura del modelo Seq2Seq y carga los parámetros necesarios.

        Args:
            hidden_size (int, optional): Tamaño del espacio oculto. Por defecto es 512.
            nb_seq (int, optional): Número máximo de secuencias. Por defecto es max_sent_size.
        """
        print("Load parameters ...")
        params = Utils.load_params(hidden_size)
        print('Loading is done.')

        # embedding
        self.embedding = npdl.layers.Embedding(params['embed_words'], nb_seq=nb_seq)
        self.embedding.connect_to()
        self.embedding.embed_words = params['embed_words']

        # encoder LSTM 1
        self.encoder_lstm1 = npdl.layers.LSTM(n_out=hidden_size, n_in=hidden_size,
                                              return_sequence=True, nb_seq=nb_seq)
        self.encoder_lstm1.connect_to(self.embedding)
        self.encoder_lstm1.U_f = params['en_lstm1']['U_f']
        self.encoder_lstm1.U_i = params['en_lstm1']['U_i']
        self.encoder_lstm1.U_o = params['en_lstm1']['U_o']
        self.encoder_lstm1.U_g = params['en_lstm1']['U_g']
        self.encoder_lstm1.W_f = params['en_lstm1']['W_f']
        self.encoder_lstm1.W_i = params['en_lstm1']['W_i']
        self.encoder_lstm1.W_o = params['en_lstm1']['W_o']
        self.encoder_lstm1.W_g = params['en_lstm1']['W_g']
        self.encoder_lstm1.b_f = params['en_lstm1']['b_f']
        self.encoder_lstm1.b_i = params['en_lstm1']['b_i']
        self.encoder_lstm1.b_o = params['en_lstm1']['b_o']
        self.encoder_lstm1.b_g = params['en_lstm1']['b_g']

        # encoder LSTM 2
        self.encoder_lstm2 = npdl.layers.LSTM(n_out=hidden_size, return_sequence=False)
        self.encoder_lstm2.connect_to(self.encoder_lstm1)
        self.encoder_lstm2.U_f = params['en_lstm2']['U_f']
        self.encoder_lstm2.U_i = params['en_lstm2']['U_i']
        self.encoder_lstm2.U_o = params['en_lstm2']['U_o']
        self.encoder_lstm2.U_g = params['en_lstm2']['U_g']
        self.encoder_lstm2.W_f = params['en_lstm2']['W_f']
        self.encoder_lstm2.W_i = params['en_lstm2']['W_i']
        self.encoder_lstm2.W_o = params['en_lstm2']['W_o']
        self.encoder_lstm2.W_g = params['en_lstm2']['W_g']
        self.encoder_lstm2.b_f = params['en_lstm2']['b_f']
        self.encoder_lstm2.b_i = params['en_lstm2']['b_i']
        self.encoder_lstm2.b_o = params['en_lstm2']['b_o']
        self.encoder_lstm2.b_g = params['en_lstm2']['b_g']

        # decoder LSTM 1
        self.decoder_lstm1 = npdl.layers.LSTM(n_out=hidden_size, n_in=hidden_size,
                                              return_sequence=True)
        self.decoder_lstm1.connect_to(self.embedding)
        self.decoder_lstm1.U_f = params['de_lstm1']['U_f']
        self.decoder_lstm1.U_i = params['de_lstm1']['U_i']
        self.decoder_lstm1.U_o = params['de_lstm1']['U_o']
        self.decoder_lstm1.U_g = params['de_lstm1']['U_g']
        self.decoder_lstm1.W_f = params['de_lstm1']['W_f']
        self.decoder_lstm1.W_i = params['de_lstm1']['W_i']
        self.decoder_lstm1.W_o = params['de_lstm1']['W_o']
        self.decoder_lstm1.W_g = params['de_lstm1']['W_g']
        self.decoder_lstm1.b_f = params['de_lstm1']['b_f']
        self.decoder_lstm1.b_i = params['de_lstm1']['b_i']
        self.decoder_lstm1.b_o = params['de_lstm1']['b_o']
        self.decoder_lstm1.b_g = params['de_lstm1']['b_g']

        # decoder LSTM 2
        self.decoder_lstm2 = npdl.layers.LSTM(n_out=hidden_size, return_sequence=False)
        self.decoder_lstm2.connect_to(self.decoder_lstm1)
        self.decoder_lstm2.U_f = params['de_lstm2']['U_f']
        self.decoder_lstm2.U_i = params['de_lstm2']['U_i']
        self.decoder_lstm2.U_o = params['de_lstm2']['U_o']
        self.decoder_lstm2.U_g = params['de_lstm2']['U_g']
        self.decoder_lstm2.W_f = params['de_lstm2']['W_f']
        self.decoder_lstm2.W_i = params['de_lstm2']['W_i']
        self.decoder_lstm2.W_o = params['de_lstm2']['W_o']
        self.decoder_lstm2.W_g = params['de_lstm2']['W_g']
        self.decoder_lstm2.b_f = params['de_lstm2']['b_f']
        self.decoder_lstm2.b_i = params['de_lstm2']['b_i']
        self.decoder_lstm2.b_o = params['de_lstm2']['b_o']
        self.decoder_lstm2.b_g = params['de_lstm2']['b_g']

        # softmax layer
        self.linear = npdl.layers.Linear(n_out=self.embedding.embed_words.shape[0])
        self.linear.connect_to(self.decoder_lstm2)
        self.linear.W = params['linear']

    def forward(self, idxs, masks):
        """
        Inicializa la arquitectura del modelo Seq2Seq y carga los parámetros necesarios.

        Args:
            hidden_size (int, optional): Tamaño del espacio oculto. Por defecto es 512.
            nb_seq (int, optional): Número máximo de secuencias. Por defecto es max_sent_size.
        """
        idxs = idxs[None, :] if np.ndim(idxs) == 1 else idxs
        masks = masks[None, :] if np.ndim(masks) == 1 else masks

        # embedding words
        embeds = self.embedding.forward(idxs)

        # encoder lstm 1
        en_lstm1_res = self.encoder_lstm1.forward(embeds, masks)

        # encoder lstm 2
        self.encoder_lstm2.forward(en_lstm1_res, masks)

        ##############################
        # Decode
        ##############################

        # functions
        g1 = self.decoder_lstm1.gate_activation.forward
        a1 = self.decoder_lstm1.activation.forward
        g2 = self.decoder_lstm2.gate_activation.forward
        a2 = self.decoder_lstm2.activation.forward

        # variables
        decodes = []
        c1_pre = self.encoder_lstm1.c0
        h1_pre = self.encoder_lstm1.h0
        c2_pre = self.encoder_lstm2.c0
        h2_pre = self.encoder_lstm2.h0
        idx = idx_start
        masks = _one((1, 1))

        # decoder LSTMs
        while True:
            # data
            idx = np.array([[idx]], dtype='int32')
            input = self.embedding.forward(idx)

            # decoder lstm 1
            de_res1 = self.decoder_lstm1.forward(input, masks, c1_pre, h1_pre)
            c1_pre, h1_pre = self.decoder_lstm1.c0, self.decoder_lstm1.h0

            # decoder lstm 2
            de_res2 = self.decoder_lstm2.forward(de_res1, masks, c2_pre, h2_pre)
            c2_pre, h2_pre = self.decoder_lstm2.c0, self.decoder_lstm2.h0

            # linear layer
            out = self.linear.forward(de_res2)
            idx = np.argmax(out[0])

            # break
            if idx == 2:
                break
            else:
                decodes.append(idx)
            if len(decodes) >= max_sent_size:
                break

        return decodes

    def utter(self, sentence):
        """
        Genera una respuesta utilizando el modelo Seq2Seq dado un texto de entrada.

        Args:
            sentence (str): Texto de entrada.

        Returns:
            str: Respuesta generada por el modelo.
        """
        idxs = np.asarray(Utils.cut_and_pad(Utils.tokens2idxs(Utils.tokenize(sentence)))[::-1], dtype='int32')
        masks = Utils.get_mask(idxs)

        idxs = self.forward(idxs, masks)

        # parse idxs to text
        tokens = Utils.idxs2tokens(idxs)
        sentence = Utils.cut_end(' '.join(tokens))
        return sentence
