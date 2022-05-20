import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import sklearn.neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
import random
import math
import dateutil.tz
import time
import datetime
import os
from . util import *

class GraphLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_heads=1, attention_dropout=0.0, activation=tf.identity, use_bias=True):
        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(hidden_size, num_heads))
        super(GraphLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.activation = activation
        self._use_bias = use_bias

    def build(self, input_shape):
        """Builds the layer."""
        size_per_head = self.hidden_size // self.num_heads

        def _glorot_initializer(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return tf.keras.initializers.RandomUniform(minval=-limit,
                                                       maxval=limit)

        w_initializer = _glorot_initializer(int(input_shape[0][-1]),
                                            size_per_head)
        self.kernels, self.biases, self.v_rows, self.v_cols= {},{},{},{}
        for i in range(self.num_heads):
            self.kernels[str(i)] = self.add_weight(name='weight_head%d'%i,
                                    shape=(int(input_shape[0][-1]),size_per_head),
                                    initializer=w_initializer,
                                    regularizer=None,
                                    trainable=True)
            if self._use_bias:
                self.biases[str(i)] = self.add_weight(name='bias_head%d'%i,
                                            shape=(size_per_head,),
                                            initializer=tf.keras.initializers.Zeros(),
                                            regularizer=None,
                                            trainable=True)
            else:
                self.biases[str(i)] = None

            self.v_rows[str(i)] = self.add_weight(name='v_row_head%d'%i,
                                    shape=(size_per_head, 1),
                                    initializer=tf.keras.initializers.GlorotUniform(),
                                    regularizer=None,
                                    trainable=True)

            self.v_cols[str(i)] = self.add_weight(name='v_col_head%d'%i,
                                    shape=(size_per_head, 1),
                                    initializer=tf.keras.initializers.GlorotUniform(),
                                    regularizer=None,
                                    trainable=True)
        
        super(GraphLayer, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }
    def call(self, inputs, training):
        """ H shape (bs, input_dim), A shape (bs, bs)
            output shape (bs, hidden_size)
        """ 
        H, adj = inputs[0], inputs[1]
        adj = tf.sparse.from_dense(adj)
        output = []
        for i in range(self.num_heads):
            H_ = tf.matmul(H, self.kernels[str(i)])
            f1 = tf.matmul(H_, self.v_rows[str(i)])
            f1 = adj * f1
            f2 = tf.matmul(H_, self.v_cols[str(i)])
            f2 = adj * tf.transpose(f2, [1, 0])
            logits = tf.compat.v1.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                         values=tf.nn.sigmoid(logits.values),
                                         dense_shape=logits.dense_shape)
            attention = tf.sparse.softmax(unnormalized_attentions)

            attention = tf.sparse.SparseTensor(indices=attention.indices,
                                         values=attention.values,
                                         dense_shape=attention.dense_shape)
            #head_output = tf.sparse.sparse_dense_matmul(attention, H_)
            head_output = tf.sparse.sparse_dense_matmul(adj, H_)
            #head_output = tf.matmul(adj, H_)
            if self._use_bias:
                head_output += self.biases[str(i)]
            if self.activation is not None:
                head_output = self.activation(head_output)
            output.append(head_output)
        output = tf.concat(output, axis=1)
        return output

class Encoder(tf.keras.Model):
    '''Encoder in SAE
    '''
    def __init__(self, params, name=None):
        super(Encoder, self).__init__(name=name)
        self.params = params
        self.all_layers = []
        for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
            if self.params['use_gcn']:
                fc1_layer = GraphLayer(n_unit)
            else:
                fc1_layer = tf.keras.layers.Dense(n_unit, 
                                activation=tf.nn.elu,
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
            bn_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc1_layer, bn_layer])
        if self.params['use_gcn']:
            fc_layer = GraphLayer(self.params['hidden_units'][-1])
        else:
            fc_layer = tf.keras.layers.Dense(self.params['hidden_units'][-1],
                                activation=tf.nn.elu,
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
        self.all_layers.append(fc_layer)
        
        #self.input_layer = tf.keras.layers.Input((self.params['dim'],))
        #self.out = self.call(self.input_layer)        

    def call(self, inputs, training=True):
        """Return the output of the Encoder.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Encoder.
            float32 tensor with shape [batch_size, hidden_dim]
        """
        if self.params['use_gcn']:
            x, adj = inputs
        else:
            x = inputs
        for i in range(len(self.params['hidden_units'])-1):
            fc1_layer, bn_layer = self.all_layers[i]
            x = fc1_layer([x, adj]) if self.params['use_gcn'] else fc1_layer(x) 
            x = tf.keras.layers.Dropout(self.params['dropout'])(x)
            x = bn_layer(x)
        fc_layer = self.all_layers[-1]
        encoded = fc_layer([x, adj]) if self.params['use_gcn'] else fc_layer(x)
        return encoded

class Decoder(tf.keras.Model):
    '''Decoder in SAE
    '''
    def __init__(self, params, name=None):
        super(Decoder, self).__init__(name=name)
        self.params = params
        self.all_layers = []
        for i, n_unit in enumerate(self.params['hidden_units'][:-1][::-1]):
            if self.params['use_gcn']:
                fc_layer = GraphLayer(n_unit)
            else:
                fc_layer = tf.keras.layers.Dense(n_unit,
                                activation=tf.nn.elu,
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
            bn_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, bn_layer])
        if self.params['use_gcn']:
            fc_layer = GraphLayer(self.params['dim'])
        else:
            fc_layer = tf.keras.layers.Dense(self.params['dim'],
                                activation=tf.nn.elu,
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
        self.all_layers.append(fc_layer)
        #self.input_layer = tf.keras.layers.Input((self.params['hidden_units'][-1],))
        #self.out = self.call(self.input_layer)    

    def call(self, inputs, training=True):
        """Return the output of the Decoder.
        Args:
            inputs: tensor with shape [batch_size, hidden_dim]
        Returns:
            Output of Decoder.
            float32 tensor with shape [batch_size, output_dim]
        """
        if self.params['use_gcn']:
            x, adj = inputs
        else:
            x = inputs
        for i in range(len(self.params['hidden_units'])-1):
            fc_layer, bn_layer = self.all_layers[i]
            x = fc_layer([x, adj]) if self.params['use_gcn'] else fc_layer(x) 
            x = tf.keras.layers.Dropout(self.params['dropout'])(x)
            x = bn_layer(x)
        decoded = self.all_layers[-1]([x,adj]) if self.params['use_gcn'] \
                    else self.all_layers[-1](x)
        return decoded

class StructuredAE(object):
    def __init__(self, params):
        super(StructuredAE, self).__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.optimizer = tf.keras.optimizers.Adam(params['lr'])
        #self.initilize_nets()
        now = datetime.datetime.now(dateutil.tz.tzlocal())

        self.timestamp = now.strftime('%Y%m%d_%H%M%S')
        self.checkpoint_path = "checkpoints/%s" % self.timestamp
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        ckpt = tf.train.Checkpoint(encoder = self.encoder,
                                   decoder = self.decoder,
                                   optimizer = self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=10)                 

        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    def get_config(self):
        return {
                "params": self.params,
        }

    def initilize_nets(self, print_summary = True):
        if self.params['use_gcn']:
            self.encoder([np.zeros((5, self.params['dim'])),np.zeros((5, 5))])
            self.decoder([np.zeros((5, self.params['hidden_units'][-1])),np.zeros((5, 5))])
        else:
            self.encoder(np.zeros((5, self.params['dim'])))
            self.decoder(np.zeros((5, self.params['hidden_units'][-1])))
        if print_summary:
            print(self.encoder.summary())
            print(self.decoder.summary())

    def get_tv_loss(self, data_x, data_x_neighbors, adj=None, adj_neighbors=None, tv_start=0, tv_end=3, use_mean=True):
        data_x = tf.cast(data_x, tf.float32)
        if self.params['use_gcn']:
            encoded = self.encoder([data_x,adj])
            z_neighbors = [self.encoder(item)[:,tv_start:tv_end] for item in zip(data_x_neighbors,adj_neighbors)]
        else:
            encoded = self.encoder(data_x)
            z_neighbors = [self.encoder(item)[:,tv_start:tv_end] for item in data_x_neighbors]
        if use_mean:
            return tf.reduce_mean([tf.reduce_mean(tf.math.abs(encoded[i,tv_start:tv_end]-z_neighbors[i])) for i in range(len(z_neighbors))])
        else:
            return [tf.reduce_mean(tf.math.abs(encoded[i,tv_start:tv_end]-z_neighbors[i])) for i in range(len(z_neighbors))]

    #Laplacian eigenmap loss
    def get_reg_loss(self, adj, encoded):
        D = tf.linalg.diag(tf.reduce_sum(adj, 1))
        L = D - adj
        reg_loss = 2*tf.linalg.trace(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(encoded),L),encoded))/encoded.shape[0]
        return reg_loss

    #graph decoder 
    def get_gd_loss(self, adj, encoded, dropout = 0.2, gd_start=0, gd_end=3):
        adj = tf.cast(adj, tf.float32)
        encoded = tf.keras.layers.Dropout(dropout)(encoded)
        rec_adj = tf.linalg.matmul(encoded[:,gd_start:gd_end], tf.transpose(encoded[:,gd_start:gd_end]))
        #rec_adj = tf.sigmoid(rec_adj)
        nb_pos = tf.reduce_sum(adj)
        nb_pos = tf.cast(nb_pos, tf.int32)
        nb_neg = tf.shape(adj)[0] ** 2
        pos_weight =  5 * nb_neg/nb_pos
        pos_weight = tf.cast(pos_weight, tf.float32)
        gd_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=adj, logits=rec_adj, pos_weight=pos_weight))
        return gd_loss

    @tf.function()
    def train_step(self, data_x, data_x_neighbors, adj, adj_neighbors, alpha, gama, tau):
        """train step.
        Args:
            inputs: input tensor list of 2
                First item:  feature tensor with shape [batch_size, input_dim].
                Second item: adjacent tensor with shape [batch_size, batch_size].
        Returns:
                returns loss functions.
        """  
        with tf.GradientTape(persistent=True) as tape:
            rec_loss, reg_loss, tv_loss, gd_loss = 0,0,0,0
            data_x = tf.cast(data_x, tf.float32)
            #AE reconstruction loss
            if self.params['use_gcn']:
                encoded = self.encoder([data_x, adj])
                #decoded = self.decoder([encoded,adj])
                decoded = self.decoder([encoded[:,50:],adj])
            else:
                encoded = self.encoder(data_x)
                #decoded = self.decoder(encoded)
                decoded = self.decoder(encoded[:,50:])
            rec_loss = tf.reduce_mean(tf.square(decoded - data_x))
            if alpha > 0:
                print('alpha works')
                reg_loss = self.get_reg_loss(adj, encoded)
            if gama > 0:
                print('gama works')
                tv_loss = self.get_tv_loss(data_x, data_x_neighbors, adj, adj_neighbors, tv_end=self.params['tv_dim'])
            if tau > 0:
                print('tau works')
                gd_loss = self.get_gd_loss(adj, encoded, gd_end=self.params['gd_dim'])
            #total_loss = rec_loss + alpha * reg_loss + gama * tv_loss + tau * gd_loss
            total_loss = gd_loss

        # Calculate the gradients
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables+self.decoder.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables+self.decoder.trainable_variables))
        return rec_loss, reg_loss, tv_loss, gd_loss, total_loss

    @tf.function
    def test_step(self, data_x, data_x_neighbors, adj):
        """train step.
        Args:
            inputs: input tensor list of 2
                First item:  feature tensor with shape [batch_size, input_dim].
                Second item: adjacent tensor with shape [batch_size, batch_size].
        Returns:
                returns loss functions.
        """
        rec_loss, reg_loss, tv_loss, gd_loss = 0,0,0,0
        data_x = tf.cast(data_x, tf.float32)
        #AE reconstruction loss 
        encoded = self.encoder(data_x)
        decoded = self.decoder(encoded)
        rec_loss = tf.reduce_mean(tf.square(decoded - data_x))
        if self.params['alpha']>0:
            reg_loss = self.get_reg_loss(adj, encoded)
        if self.params['gama']>0:
            tv_loss = self.get_tv_loss(data_x, data_x_neighbors, tv_end=self.params['tv_dim'])
        if self.params['tau']>0:
            gd_loss = self.get_gd_loss(adj, encoded, gd_end=self.params['gd_dim'])
        total_loss = rec_loss + self.params['alpha']*reg_loss + self.params['gama']*tv_loss + self.params['tau']*gd_loss
        return rec_loss, reg_loss, tv_loss, gd_loss, total_loss


    def fit(self, X, adj, adj_indices, bootstrap=False, eval_every=5, val_split=0., patience=10, random_seed=2022):
        bs = self.params['batch_size']
        alpha, gama, tau = self.params['alpha'],self.params['gama'],self.params['tau']
        np.random.seed(random_seed)
        random.seed(random_seed)
        sample_indx = np.arange(X.shape[0])
        f_log = open('%s/log.txt'%self.checkpoint_path,'a+')
        #no validation split
        if val_split == 0:
            np.random.shuffle(sample_indx)
            train_indx = sample_indx
            for epoch in range(self.params['max_epochs']):
                #if epoch >= 80:
                #    gama = 0.1
                train_rec_loss_list, train_reg_loss_list, train_tv_loss_list, train_gd_loss_list, train_total_loss_list = [],[],[],[],[]
                t0 = time.time()
                #fit training data
                for batch_idx in range(len(train_indx)//bs):
                    if bootstrap:
                        center_idx = random.randint(0,X.shape[0]-1)
                        indx = adj_list[center_idx][:bs] #adj_list records batch neighbors
                    else:
                        indx = train_indx[batch_idx*bs:(batch_idx+1)*bs]
                    X_batch = X[indx,:]
                    #TODO handle sparse adj
                    adj_batch = adj[indx,:][:,indx]
                    X_neighbors_batch = [X[adj_indices[i]] for i in indx]
                    adj_neighbors_batch = [adj[adj_indices[i],:][:,adj_indices[i]] for i in indx]
                    rec_loss, reg_loss, tv_loss, gd_loss, total_loss = self.train_step(X_batch,X_neighbors_batch, \
                                            adj_batch,adj_neighbors_batch, alpha, gama, tau)
                    train_rec_loss_list += [rec_loss]
                    train_reg_loss_list += [reg_loss]
                    train_tv_loss_list += [tv_loss]
                    train_gd_loss_list += [gd_loss]
                    train_total_loss_list += [total_loss]
                t = time.time()-t0
                print('Epoch [%d] within %.2f seconds: train_rec_loss [%.4f], train_reg_loss [%.4f], train_tv_loss [%.4f], '\
                    'train_gd_loss [%.4f], train_total_loss [%.4f]'%
                    (epoch, t, np.mean(train_rec_loss_list), np.mean(train_reg_loss_list), np.mean(train_tv_loss_list), 
                    np.mean(train_gd_loss_list), np.mean(train_total_loss_list)))
                if epoch % eval_every==0:
                    ari = self.evaluate(X, adj, adj_indices, epoch)
                    contents = '''Epoch [%d] ARI [%.4f] within %.2f seconds: train_rec_loss [%.4f], train_reg_loss [%.4f],\
                    train_tv_loss [%.4f], train_gd_loss [%.4f], train_total_loss [%.4f]\n''' \
                    %(epoch, ari, t, np.mean(train_rec_loss_list), np.mean(train_reg_loss_list), np.mean(train_tv_loss_list),
                    np.mean(train_gd_loss_list), np.mean(train_total_loss_list))
                    f_log.write(contents)
        else:
            train_indx, val_indx = train_test_split(sample_indx, test_size=val_split, random_state=42)
            best_val_loss = np.inf
            wait = 0
            for epoch in range(self.params['max_epochs']):
                train_rec_loss_list, train_reg_loss_list, train_tv_loss_list, train_gd_loss_list, train_total_loss_list = [],[],[],[],[]
                t0 = time.time()
                #fit training data
                for batch_idx in range(len(train_indx)//bs):
                    indx = train_indx[batch_idx*bs:(batch_idx+1)*bs]
                    X_batch = X[indx,:]
                    #TODO handle sparse adj
                    adj_batch = adj[indx,:][:,indx]
                    X_neighbors_batch = [X[adj_indices[i]] for i in indx]
                    rec_loss, reg_loss, tv_loss, gd_loss, total_loss = self.train_step(X_batch,X_neighbors_batch,adj_batch)
                    train_rec_loss_list += [rec_loss]
                    train_reg_loss_list += [reg_loss]
                    train_tv_loss_list += [tv_loss]
                    train_gd_loss_list += [gd_loss]
                    train_total_loss_list += [total_loss]

                #cal validation loss
                val_rec_loss_list, val_reg_loss_list, val_tv_loss_list, val_gd_loss_list, val_total_loss_list = [],[],[],[],[]
                for batch_idx in range(len(val_indx)//bs):
                    indx = val_indx[batch_idx*bs:(batch_idx+1)*bs]
                    X_batch = X[indx,:]
                    #TODO handle sparse adj
                    adj_batch = adj[indx,:][:,indx]
                    X_neighbors_batch = [X[adj_indices[i]] for i in indx]
                    rec_loss, reg_loss, tv_loss, gd_loss, total_loss = self.test_step(X_batch,X_neighbors_batch,adj_batch)
                    val_rec_loss_list += [rec_loss]
                    val_reg_loss_list += [reg_loss]
                    val_tv_loss_list += [tv_loss]
                    val_gd_loss_list += [gd_loss]
                    val_total_loss_list += [total_loss]  
                t = time.time()-t0
                print('Epoch [%d] within %.2f seconds: train_rec_loss [%.4f], train_reg_loss [%.4f], train_tv_loss [%.4f], '\
                    'train_gd_loss [%.4f], train_total_loss [%.4f], val_rec_loss [%.4f], val_reg_loss [%.4f], val_tv_loss [%.4f], '\
                    'val_gd_loss [%.4f], val_total_loss [%.4f]'%
                    (epoch, t, np.mean(train_rec_loss_list), np.mean(train_reg_loss_list), np.mean(train_tv_loss_list), 
                    np.mean(train_gd_loss_list), np.mean(train_total_loss_list), np.mean(val_rec_loss_list), np.mean(val_reg_loss_list),
                    np.mean(val_tv_loss_list), np.mean(val_gd_loss_list), np.mean(val_total_loss_list)))
                if epoch % eval_every==0:
                    ari = self.evaluate(X, adj, adj_indices, epoch)
                    contents = '''Epoch [%d] ARI [%.4f] within %.2f seconds: train_rec_loss [%.4f], train_reg_loss [%.4f], \
                    train_tv_loss [%.4f], train_gd_loss [%.4f], train_total_loss [%.4f], val_rec_loss [%.4f], val_reg_loss [%.4f], \
                    val_tv_loss [%.4f], val_gd_loss [%.4f], val_total_loss [%.4f]\n''' \
                    %(epoch, ari, t, np.mean(train_rec_loss_list), np.mean(train_reg_loss_list), np.mean(train_tv_loss_list),
                    np.mean(train_gd_loss_list), np.mean(train_total_loss_list), np.mean(val_rec_loss_list), np.mean(val_reg_loss_list),
                    np.mean(val_tv_loss_list), np.mean(val_gd_loss_list), np.mean(val_total_loss_list))
                    f_log.write(contents)
                if np.mean(val_total_loss_list) < best_val_loss:
                    wait = 0
                    best_val_loss = np.mean(val_total_loss_list)
                else:
                    wait += 1
                    if wait >= patience or epoch==self.params['max_epochs']:
                        print('Early stopping at epoch [%d]!'%epoch)
                        break
        f_log.close()

    def predict(self, X, adj):
        if self.params['use_gcn']:
            return self.encoder([X,adj]).numpy()
        else:
            return self.encoder(X).numpy()

    def evaluate(self, X, adj, adj_indices, epoch, save=True, method='Louvain'):
        ae_embeds = self.predict(X, adj)
        #calculate TV L1 norm 
        neighbors_all = [X[indx] for indx in adj_indices]
        #adj_neighbors_all = [adj[adj_indices[i],:][:,adj_indices[i]] for i in range(X.shape[0])]
        #tv_loss = self.get_tv_loss(X, neighbors_all, adj, adj_neighbors_all, use_mean=False)
        #bg_tv_loss = self.get_tv_loss(X, neighbors_all, tv_start=self.params['hidden_units'][-1]-self.params['tv_dim'], tv_end=self.params['hidden_units'][-1], use_mean=False)
        if save:
            np.save('%s/ae_embeds_%d.npy'%(self.checkpoint_path, epoch), ae_embeds)
            #np.save('%s/tv_loss_%d.npy'%(self.checkpoint_path, epoch), np.vstack([tv_loss,bg_tv_loss]))
            #np.save('%s/tv_loss_%d.npy'%(self.checkpoint_path, epoch), tv_loss)
            ckpt_save_path = self.ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch,ckpt_save_path))
        if 'annotation' in self.params:
            if method=='mclust':
                adata=sc.AnnData(X)
                y_pred = mclust_clustering(X=ae_embeds, K=self.params['nb_clusters'])
            elif method=='Louvain':
                y_pred = louvain_clustering(X=ae_embeds, K=self.params['nb_clusters'])
            else:
                print('Specify clustering algorithm as either mclust or Louvain')
            y_annot = self.params['annotation'].to_list()
            ari = adjusted_rand_score(y_annot, y_pred)
            print('Epoch [%d] ARI [%.4f]'%(epoch, ari))
            return ari
