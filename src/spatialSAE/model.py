import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import dateutil.tz
import datetime
import os

class Encoder(tf.keras.Model):
    '''Encoder in SAE
    '''
    def __init__(self, params, name=None):
        super(Encoder, self).__init__(name=name)
        self.params = params
        self.all_layers = []
        if self.params['use_resnet']:
            for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
                fc1_layer = tf.keras.layers.Dense(n_unit, 
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
                bn_layer = tf.keras.layers.BatchNormalization()
                fc2_layer = tf.keras.layers.Dense(n_unit, 
                                activation=None,
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
                self.all_layers.append([fc1_layer, bn_layer, fc2_layer])
            fc_layer = tf.keras.layers.Dense(self.params['hidden_units'][-1],
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
            self.all_layers.append(fc_layer)
        else:
            for i, n_unit in enumerate(self.params['hidden_units'][:-1]):
                fc1_layer = tf.keras.layers.Dense(n_unit, 
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
                bn_layer = tf.keras.layers.BatchNormalization()
                self.all_layers.append([fc1_layer, bn_layer])
            fc_layer = tf.keras.layers.Dense(self.params['hidden_units'][-1],
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
            self.all_layers.append(fc_layer)
        self.input_layer = tf.keras.layers.Input((self.params['dim'],))
        self.out = self.call(self.input_layer)        


    def call(self, inputs, training=True):
        """Return the output of the Encoder.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Encoder.
            float32 tensor with shape [batch_size, hidden_dim]
        """
        if self.params['use_resnet']:
            for i in range(len(self.params['hidden_units'])-1):
                fc1_layer, bn_layer, fc2_layer = self.all_layers[i]
                x_init = fc1_layer(inputs) if i==0 else fc1_layer(x) 
                x = tf.keras.layers.Dropout(0.1)(x_init)
                x = x_init
                x = bn_layer(x)
                x = fc2_layer(x)
                x = tf.keras.layers.Add()([x,x_init])
                x = tf.keras.layers.Activation(activation='relu')(x)
            fc_layer = self.all_layers[-1]
            encoded = fc_layer(x)
        else:
            for i in range(len(self.params['hidden_units'])-1):
                fc1_layer, bn_layer = self.all_layers[i]
                x = fc1_layer(inputs) if i==0 else fc1_layer(x) 
                x = tf.keras.layers.Dropout(0.2)(x)
                x = bn_layer(x)
            fc_layer = self.all_layers[-1]
            encoded = fc_layer(x)
        return encoded

class Decoder(tf.keras.Model):
    '''Decoder in SAE
    '''
    def __init__(self, params, name=None):
        super(Decoder, self).__init__(name=name)
        self.params = params
        self.all_layers = []
        for i, n_unit in enumerate(self.params['hidden_units'][:-1][::-1]):
            fc_layer = tf.keras.layers.Dense(n_unit,
                                activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                                bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
            bn_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, bn_layer])
        fc_layer = tf.keras.layers.Dense(self.params['dim'],
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.L2(self.params['beta']),
                            bias_regularizer=tf.keras.regularizers.L2(self.params['beta']))
        self.all_layers.append(fc_layer)
        self.input_layer = tf.keras.layers.Input((self.params['hidden_units'][-1],))
        self.out = self.call(self.input_layer)    

    def call(self, inputs, training=True):
        """Return the output of the Decoder.
        Args:
            inputs: tensor with shape [batch_size, hidden_dim]
        Returns:
            Output of Decoder.
            float32 tensor with shape [batch_size, output_dim]
        """
        for i in range(len(self.params['hidden_units'])-1):
            fc_layer, bn_layer = self.all_layers[i]
            x = fc_layer(inputs) if i==0 else fc_layer(x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            x = bn_layer(x)
        decoded = self.all_layers[-1](x)
        return decoded

class StructuredAE(object):
    def __init__(self, params):
        super(StructuredAE, self).__init__()
        self.params = params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.optimizer = tf.keras.optimizers.Adam(params['lr'])
        self.initilize_nets()
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
        self.encoder(np.zeros((5, self.params['dim'])))
        self.decoder(np.zeros((5, self.params['hidden_units'][-1])))
        if print_summary:
            print(self.encoder.summary())
            print(self.decoder.summary())

    @tf.function
    def train_step(self, data_x, adj):
        """train step.
        Args:
            inputs: input tensor list of 2
                First item:  feature tensor with shape [batch_size, input_dim].
                Second item: adjacent tensor with shape [batch_size, batch_size].
        Returns:
                returns loss functions.
        """  
        with tf.GradientTape(persistent=True) as tape:
            data_x = tf.cast(data_x, tf.float32)
            encoded = self.encoder(data_x)
            decoded = self.decoder(encoded)
            rec_loss = tf.reduce_mean(tf.square(decoded - data_x))
            D = tf.linalg.diag(tf.reduce_sum(adj, 1))
            L = D - adj #Laplacian matrix
            reg_loss = 2*tf.linalg.trace(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(encoded),L),encoded))/encoded.shape[0]
            total_loss = rec_loss+self.params['alpha']*reg_loss

        # Calculate the gradients
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables+self.decoder.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables+self.decoder.trainable_variables))
        return rec_loss, reg_loss, total_loss

    @tf.function
    def test_step(self, data_x, adj):
        data_x = tf.cast(data_x, tf.float32)
        encoded = self.encoder(data_x)
        decoded = self.decoder(encoded)
        rec_loss = tf.reduce_mean(tf.square(decoded - data_x))
        D = tf.linalg.diag(tf.reduce_sum(adj, 1))
        L = D - adj #Laplacian matrix
        reg_loss = 2*tf.linalg.trace(tf.linalg.matmul(tf.linalg.matmul(tf.transpose(encoded),L),encoded))/encoded.shape[0]
        return rec_loss, reg_loss, total_loss

    def fit(self, X, adj, bs=64, max_epochs=200, save_every=20, val_split=0.05, patience=5):
        indx = np.arange(X.shape[0])
        np.random.shuffle(indx)
        train_indx, val_indx = train_test_split(indx, test_size=val_split, random_state=42)
        best_val_loss = np.inf
        wait = 0
        for epoch in range(max_epochs):
            train_rec_loss_list, train_reg_loss_list, train_total_loss_list = [], [] ,[] 
            #fit training data
            for batch_idx in range(len(train_indx)//bs):
                indx = train_indx[batch_idx*bs:(batch_idx+1)*bs]
                X_batch = X[indx,:]
                #TODO handle sparse adj
                adj_batch = adj[indx,:][:,indx]
                rec_loss, reg_loss, total_loss = self.train_step(X_batch,adj_batch)
                train_rec_loss_list += [rec_loss]
                train_reg_loss_list += [reg_loss]
                train_total_loss_list += [total_loss]

            #cal validation loss
            val_rec_loss_list, val_reg_loss_list, val_total_loss_list = [], [] ,[] 
            for batch_idx in range(len(val_indx)//bs):
                indx = val_indx[batch_idx*bs:(batch_idx+1)*bs]
                X_batch = X[indx,:]
                #TODO handle sparse adj
                adj_batch = adj[indx,:][:,indx]
                rec_loss, reg_loss, total_loss = self.train_step(X_batch,adj_batch)
                val_rec_loss_list += [rec_loss]
                val_reg_loss_list += [reg_loss]
                val_total_loss_list += [total_loss]            
            
            print('Epoch [%d]: train_rec_loss [%.4f], train_reg_loss [%.4f], train_total_loss [%.4f], '\
                'val_rec_loss [%.4f], val_reg_loss [%.4f], val_total_loss [%.4f]'%
                (epoch, np.mean(train_rec_loss_list), np.mean(train_reg_loss_list), np.mean(train_total_loss_list),
                np.mean(val_rec_loss_list), np.mean(val_reg_loss_list), np.mean(val_total_loss_list)))
            if epoch % save_every == 0:
                ae_embeds = self.predict(X)
                np.save('%s/ae_embeds_%d.npy'%(self.checkpoint_path, epoch), ae_embeds)
                ckpt_save_path = self.ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch,ckpt_save_path))
            if np.mean(val_total_loss_list) < best_val_loss:
                wait = 0
                best_val_loss = np.mean(val_total_loss_list)
            else:
                wait += 1
                if wait >= patience:
                    print('Early stopping at epoch [%d]!'%epoch)
                    break

    def predict(self, X):
        return self.encoder(X)
