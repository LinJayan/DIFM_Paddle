# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from paddle.regularizer import L2Decay


class DIFMLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes_dnn,dense_layers_size, num_heads,qkv_dim):
        super(DIFMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes_dnn = layer_sizes_dnn
        self.dense_layers_size = dense_layers_size

        self.linear = LinearModel(sparse_feature_number, sparse_feature_dim,
                     dense_feature_dim, sparse_num_field,dense_layers_size)

        self.dual_fen = Dual_FENLayer(sparse_feature_number, sparse_feature_dim,
                 sparse_num_field,layer_sizes_dnn,num_heads,qkv_dim)

        self.att = InteractingLayer(sparse_feature_dim)
        self.fm = FM()
        self.dnn = DNNLayer(sparse_num_field * sparse_feature_dim) 


    def forward(self, sparse_inputs, dense_inputs):

        # # Dual_FENLayer
        m_vec, m_bit = self.dual_fen(sparse_inputs)

        dnn_logits, feat_emb_one, feat_embeddings = self.linear(sparse_inputs, dense_inputs)

        # Combination Layer
        input_aware_factor = paddle.add(m_vec,m_bit)
        input_aware_factor = F.softmax(input_aware_factor)

        # Reweighting Layer
        m_x = F.softmax(m_vec + m_bit)

        feat_emb_one = feat_emb_one * m_x
        feat_embeddings = feat_embeddings * paddle.unsqueeze(m_x, axis=-1)

        first_order = paddle.sum(feat_emb_one, axis=1, keepdim=True)

        # PredictionLayer
        fm_out = self.fm(dnn_logits, first_order, feat_embeddings)

        return fm_out


class Dual_FENLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 sparse_num_field,layer_sizes_dnn,num_heads,qkv_dim):
        super(Dual_FENLayer,self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes_dnn = layer_sizes_dnn
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim
       

        self.init_value_ = 0.1

        self.multi_att = InteractingLayer(sparse_feature_dim, num_heads, qkv_dim)

        self.dnn = DNNLayer(sparse_num_field * sparse_feature_dim,layer_sizes_dnn) 

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))),
                # regularizer=paddle.regularizer.L2Decay(1e-6)
                )) 


        self.p_vec = paddle.nn.Linear(
                            in_features=self.qkv_dim * self.sparse_num_field,
                            out_features=self.sparse_num_field+1,
                            weight_attr=paddle.ParamAttr(
                            regularizer=L2Decay(coeff=1e-6),
                            initializer=paddle.nn.initializer.Normal(
                            std=0.1 / math.sqrt(self.qkv_dim * self.sparse_num_field))))

        self.p_bit = paddle.nn.Linear(
                            in_features=self.layer_sizes_dnn[-1],
                            out_features=self.sparse_num_field+1,
                            weight_attr=paddle.ParamAttr(
                            regularizer=L2Decay(coeff=1e-6),
                            initializer=paddle.nn.initializer.Normal(
                            std=0.1 / math.sqrt(self.sparse_feature_dim * self.sparse_num_field))))


        self.relu = paddle.nn.ReLU()
        self.act = paddle.nn.Softmax()


    def forward(self, inputs):

        # EmbeddingLayer
        sparse_inputs_concat = paddle.concat(inputs, axis=1)
        sparse_embeddings = self.embedding(sparse_inputs_concat) # [batch_size,sparse_num_field,dim]

        # The vector-wise part
        att_out = self.multi_att(sparse_embeddings)
        att_out = paddle.nn.Flatten()(att_out)
        m_vec = self.p_vec(att_out)
        m_vec = self.relu(m_vec)
        # m_vec = self.act(m_vec)

        # The bit-wise part
        dnn_out = self.dnn(sparse_embeddings)
        m_bit = self.p_bit(dnn_out) 
        
        m_bit = self.act(m_bit)

        return m_vec, m_bit

class LinearModel(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field,dense_layers_size):
        super(LinearModel, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1
        self.dense_layers_size = dense_layers_size

        self.dense_linear = paddle.nn.Linear(in_features=self.dense_feature_dim,
                                             out_features=1)

        self.dense_mlp = MLPLayer(input_shape=self.dense_feature_dim,
                                  units_list=dense_layers_size,
                                  last_action="relu")
        self.dnn_mlp = MLPLayer(input_shape=self.sparse_feature_dim,
                                units_list=[1],
                                last_action="relu")

        self.embedding = paddle.nn.Embedding(num_embeddings=self.sparse_feature_number,
                                                    embedding_dim=sparse_feature_dim,
                                                    sparse=True,
                                                    weight_attr=paddle.ParamAttr(
                                                        name="Sparse1",
                                                        initializer=paddle.nn.initializer.Uniform()))

        self.embedding_one = paddle.nn.Embedding(num_embeddings=self.sparse_feature_number,
                                                 embedding_dim=1,
                                                 sparse=True,
                                                 weight_attr=paddle.ParamAttr(
                                                     name="Sparse2",
                                                     initializer=paddle.nn.initializer.Uniform()))

        # dense coding
        self.dense_w_one = paddle.create_parameter(
            shape=[self.dense_feature_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))


    def forward(self, sparse_inputs, dense_inputs):

        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)
        sparse_emb_one = paddle.squeeze(sparse_emb_one, axis=-1)
        dense_emb_one = self.dense_linear(dense_inputs)

        # -------------------- first order term  --------------------
        feat_emb_one = paddle.concat([dense_emb_one, sparse_emb_one], axis=1)

        dense_embedding = self.dense_mlp(dense_inputs)
        dnn_logits = self.dnn_mlp(dense_embedding)
        dense_embedding = paddle.unsqueeze(dense_embedding, axis=1)

        sparse_embedding = self.embedding(sparse_inputs_concat)

        feat_embeddings = paddle.concat([dense_embedding, sparse_embedding], axis=1)

        return dnn_logits, feat_emb_one, feat_embeddings

class FM(nn.Layer):
    def __init__(self):
        super(FM, self).__init__()
        self.bias = paddle.create_parameter(is_bias=True,
                                            shape=[1],
                                            dtype='float32')

    def forward(self, dnn_logits, first_order, combined_features):
    
        # sum square part
        # (batch_size, embedding_size)
        summed_features_emb = paddle.sum(combined_features, axis=1)
        summed_features_emb_square = paddle.square(summed_features_emb)

        # square sum part
        squared_features_emb = paddle.square(combined_features)
        # (batch_size, embedding_size)
        squared_sum_features_emb = paddle.sum(squared_features_emb, axis=1)

        # (batch_size, 1)
        logits = first_order + 0.5 * paddle.sum(summed_features_emb_square - squared_sum_features_emb, axis=1,
                                                keepdim=True) + self.bias + dnn_logits
        return F.sigmoid(logits)

class MLPLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, l2=0.01, last_action=None, **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [128, 128, 64]
        units_list = [input_shape] + units_list

        self.units_list = units_list
        self.l2 = l2
        self.mlp = []
        self.last_action = last_action

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                             initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit)))
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                if last_action is not None:
                    relu = paddle.nn.ReLU()
                    self.mlp.append(relu)
                    self.add_sublayer('relu_%d' % i, relu)

    def forward(self, inputs):
        outputs = inputs
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
        return outputs

class DNNLayer(nn.Layer):
    def __init__(self, input_shape, units_list=None, dropout_rate=0.5, **kwargs):
        super(DNNLayer, self).__init__(**kwargs)

        if units_list is None:
            units_list = [256, 256, 256]
        units_list = [input_shape] + units_list

        self.units_list = units_list
       
        self.mlp = []

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)
    

        for i, unit in enumerate(units_list[:-1]):
            if i != len(units_list) - 1:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                             regularizer=paddle.regularizer.L2Decay(1e-6),
                                             initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))  # 1.0
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

                norm = paddle.nn.BatchNorm1D(units_list[i + 1])
                self.mlp.append(norm)
                self.add_sublayer('norm_%d' % i, norm)
            else:
                dense = paddle.nn.Linear(in_features=unit,
                                         out_features=units_list[i + 1],
                                         weight_attr=paddle.ParamAttr(
                                            regularizer=paddle.regularizer.L2Decay(1e-6),
                                            initializer=paddle.nn.initializer.Normal(std=1.0 / math.sqrt(unit))))  # 1.0
                self.mlp.append(dense)
                self.add_sublayer('dense_%d' % i, dense)

                relu = paddle.nn.ReLU()
                self.mlp.append(relu)
                self.add_sublayer('relu_%d' % i, relu)

    def get_shape(self,inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        in_shape1, in_shape2 = inputs.shape[1],inputs.shape[2]
        return in_shape1, in_shape2

    def forward(self, inputs):
        num_field, sparse_feature_dim = self.get_shape(inputs)

        y_dnn = paddle.reshape(inputs,
                               [-1, num_field * sparse_feature_dim])

        outputs = y_dnn
        for n_layer in self.mlp:
            outputs = n_layer(outputs)
            outputs = self.drop_out(outputs)
        return outputs



class InteractingLayer(paddle.nn.Layer):
    def __init__(self, embedding_size, head_num=12, qkv_dim=60, use_res=True, scaling=True):
        super(InteractingLayer, self).__init__()

        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if qkv_dim % head_num != 0:
            raise ValueError('qkv_dim is not an integer multiple of head_num!')
        self.att_embedding_size = qkv_dim // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.qkv_dim = qkv_dim

        self.init_value_ = 0.1

        self.W_q = paddle.nn.Linear(
                            in_features=embedding_size,
                            out_features=self.qkv_dim,
                            weight_attr=paddle.ParamAttr(
                            # regularizer=L2Decay(coeff=1e-6),
                            initializer=paddle.nn.initializer.Normal(
                            std=0.1 / math.sqrt(embedding_size * self.qkv_dim))))
        self.W_k = paddle.nn.Linear(
                            in_features=embedding_size,
                            out_features=self.qkv_dim,
                            weight_attr=paddle.ParamAttr(
                            # regularizer=L2Decay(coeff=1e-6),
                            initializer=paddle.nn.initializer.Normal(
                            std=0.1 / math.sqrt(embedding_size * self.qkv_dim))))
        self.W_v = paddle.nn.Linear(
                            in_features=embedding_size,
                            out_features=self.qkv_dim,
                            weight_attr=paddle.ParamAttr(
                            # regularizer=L2Decay(coeff=1e-6),
                            initializer=paddle.nn.initializer.Normal(
                            std=0.1 / math.sqrt(embedding_size * self.qkv_dim))))

        self.W_Query = paddle.create_parameter(
                                        shape=[self.qkv_dim],
                                        dtype='float32',
                                        default_initializer=paddle.nn.initializer.TruncatedNormal(
                                        mean=0.0,
                                        std=self.init_value_ /
                                        math.sqrt(float(self.qkv_dim))))
        self.W_key = paddle.create_parameter(
                                        shape=[self.qkv_dim],
                                        dtype='float32',
                                        default_initializer=paddle.nn.initializer.TruncatedNormal(
                                        mean=0.0,
                                        std=self.init_value_ /
                                        math.sqrt(float(self.qkv_dim))))
        self.W_Value = paddle.create_parameter(
                                        shape=[self.qkv_dim],
                                        dtype='float32',
                                        default_initializer=paddle.nn.initializer.TruncatedNormal(
                                        mean=0.0,
                                        std=self.init_value_ /
                                        math.sqrt(float(self.qkv_dim))))

        if self.use_res:
            self.W_Res = paddle.create_parameter(
                                        shape=[self.qkv_dim],
                                        dtype='float32',
                                        default_initializer=paddle.nn.initializer.TruncatedNormal(
                                        mean=0.0,
                                        std=self.init_value_ /
                                        math.sqrt(float(self.qkv_dim))))

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # pre q k v
        q = self.W_q(inputs)
        k = self.W_k(inputs)
        v = self.W_v(inputs)

        # None F D
        querys = paddle.multiply(q, self.W_Query)
        keys = paddle.multiply(k, self.W_key)
        values = paddle.multiply(v, self.W_Value)

        # head_num None F D/head_num
        querys = paddle.stack(paddle.split(querys, self.att_embedding_size,  axis=2))
        keys = paddle.stack(paddle.split(keys, self.att_embedding_size, axis=2))
        values = paddle.stack(paddle.split(values, self.att_embedding_size, axis=2))
        # print(querys.shape)

        inner_product = paddle.matmul( x=querys, y=keys, transpose_y=True)   # head_num None F F
        # print(inner_product.shape)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        normalized_att_scores = F.softmax(inner_product, axis=-1)  # head_num None F F
        # print(normalized_att_scores.shape)

        result = paddle.matmul(normalized_att_scores, values)  # head_num None F D/head_num

        result = paddle.concat(paddle.split(result, self.att_embedding_size,axis=0), axis=-1)
        
        result = paddle.squeeze(result, axis=0)  # None F D
        
        if self.use_res:
            result += paddle.multiply(v, self.W_Res)
        result = F.relu(result)
        
        return result




