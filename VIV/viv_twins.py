"""VIV model on Twins"""
import os, tensorflow as tf
from tensorflow.contrib import slim
from progressbar import ETA, Bar, Percentage, ProgressBar
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.distributions as tfd, numpy as np, time, sys
import scipy

def log(logfile, str, out=True):
    """ Log a string in a file """
    with open(logfile, 'a') as (f):
        f.write(str + '\n')
    if out:
        print(str)


def get_FLAGS():
    """ Define parameter flags """
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('earl', 10, 'when to show bound ')
    tf.app.flags.DEFINE_float('lrate', 0.0001, 'Learning rate. ')
    tf.app.flags.DEFINE_float('lrate_min', 0.001, 'Learning rate min. ')
    tf.app.flags.DEFINE_integer('epochs', 10, 'epochs ')
    tf.app.flags.DEFINE_integer('seed', 2023, 'Seed. ')
    tf.app.flags.DEFINE_integer('bs', 256, 'Batch size. ')
    tf.app.flags.DEFINE_integer('d', 2, 'Latent dimension. ')
    tf.app.flags.DEFINE_boolean('rewrite_log', 0, 'Whether rewrite log file. ')
    tf.app.flags.DEFINE_boolean('use_gpu', 1, 'The use of GPU. ')
    tf.app.flags.DEFINE_float('lamba', 0.0001, 'weight decay. ')
    tf.app.flags.DEFINE_integer('nh', 3, 'number of hidden layers. ')
    tf.app.flags.DEFINE_integer('h', 128, 'size of hidden layers. ')
    tf.app.flags.DEFINE_integer('reps', 10, 'replications. ')
    tf.app.flags.DEFINE_string('f', '', 'kernel')
    tf.app.flags.DEFINE_string('activation', 'elu', 'activation function leaky_relu')
    tf.app.flags.DEFINE_float('loss_y', 0.1, 'loss y')
    tf.app.flags.DEFINE_float('loss_t', 0.1, 'loss t')
    tf.app.flags.DEFINE_float('loss_x', 0.1, 'loss x')
    tf.app.flags.DEFINE_float('kl_loss', 0.1, 'kl loss')
    tf.app.flags.DEFINE_float('ad_loss', 1, 'adversarial loss')
    tf.app.flags.DEFINE_integer('lrate_decay_num', 10, 'NUM_ITERATIONS_PER_DECAY. ')
    tf.app.flags.DEFINE_float('lrate_decay', 0.97, 'Decay of learning rate every 100 iterations ')
    tf.app.flags.DEFINE_float('decay', 0.97, 'Decay of learning rate every 100 iterations ')
    tf.app.flags.DEFINE_string('optimizer', 'RMSProp', 'Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)')
    tf.app.flags.DEFINE_integer('output_delay', 100, 'Number of iterations between log/loss outputs. ')
    if FLAGS.sparse:
        import scipy.sparse as sparse
    return FLAGS


class VIV(object):

    def __init__(self, x_1, x_2, y, t, args, q_labels):
        self.x_1 = x_1
        self.x_2 = x_2
        self.x_1_one_hot = tf.one_hot(indices=(self.x_1), depth=5, axis=(-1))
        self.x_2_one_hot = tf.one_hot(indices=(self.x_2), depth=4, axis=(-1))
        self.x = tf.concat([self.x_1_one_hot, self.x_2_one_hot], 1)
        self.y = y
        self.t = t
        self.q_labels = q_labels
        self.build_graph(args)

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * tf.reduce_mean(tf.reduce_sum((1 + logvar - tf.square(mu) - tf.exp(logvar)),-1))
        return kld
    def wassertein_distance(self,mu,logvar):
        p1 = tf.reduce_mean(tf.reduce_sum(tf.square((mu)),-1))
        p2 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sqrt(tf.exp(logvar))-1),-1))
        return p1+p2
    def diagonal(self, M):
        new_M = tf.where(tf.abs(M) < 1e-05, M + 1e-05 * tf.abs(M), M)
        return new_M

    def reparameterize(self, mu, logvar):
        noise_ = tf.random.normal(shape=(tf.shape(mu)))
        output = mu + noise_ * tf.exp(0.5 * logvar)
        return output

    def permute_dims(self, u, d):
        B = tf.shape(u)[0]
        id_ = tf.range(B)
        perm_u = []
        for u_j in tf.split(u, [d, d, d,d], 1):
            id_perm = tf.random.shuffle(id_)
            perm_u_j = tf.gather(u_j, id_perm)
            perm_u.append(perm_u_j)

        return tf.concat(perm_u, 1)

    def fc_net(self, inp, layers, out_layers, scope, lamba=0.001, activation=tf.nn.relu, reuse=None, weights_initializer=initializers.xavier_initializer(uniform=False)):
        with slim.arg_scope([slim.fully_connected], activation_fn=activation,
          normalizer_fn=None,
          weights_initializer=weights_initializer,
          reuse=reuse,
          weights_regularizer=(slim.l2_regularizer(lamba))):
            if layers:
                h = slim.stack(inp, (slim.fully_connected), layers, scope=scope)
                if not out_layers:
                    return h
            else:
                h = inp
            outputs = []
            for i, (outdim, activation) in enumerate(out_layers):
                o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=(scope + '_{}'.format(i + 1)))
                outputs.append(o1)

            if len(outputs) > 1:
                return outputs
            else:
                return outputs[0]

    def build_graph(self, args):
        """VIV variational approximation (encoder)"""
        if args.activation == 'elu':
            activation = tf.nn.elu
        else:
            if args.activation == 'relu':
                activation = tf.nn.relu
            if args.activation == 'leaky_relu':
                activation = tf.nn.leaky_relu
        inptz = tf.concat([self.t], 1)
        muq_z, sigmaq_z = self.fc_net(inptz, ((args.nh - 1) * [args.h]), [[args.d, None], [args.d, None]], 'qz_t', lamba=(args.lamba), activation=activation)
        self.qz = self.reparameterize(muq_z, sigmaq_z)
        inptc = tf.concat([self.x, self.t, self.y], 1)
        muq_c, sigmaq_c = self.fc_net(inptc, ((args.nh - 1) * [args.h]), [[args.d, None], [args.d, None]], 'qc_xty', lamba=(args.lamba), activation=activation)
        self.qc = self.reparameterize(muq_c, sigmaq_c)
        inpta = tf.concat([self.x, self.y], 1)
        muq_a, sigmaq_a = self.fc_net(inpta, ((args.nh - 1) * [args.h]), [[args.d, None], [args.d, None]], 'qa_xy', lamba=(args.lamba), activation=activation)
        self.qa = self.reparameterize(muq_a, sigmaq_a)
        inptu = tf.concat([self.t, self.y], 1)
        muq_u, sigmaq_u = self.fc_net(inptu, ((args.nh - 1) * [args.h]), [[args.d, None], [args.d, None]], 'qu_ty', lamba=(args.lamba), activation=activation)
        self.qu = self.reparameterize(muq_u, sigmaq_u)
        """VIV variational approximation (decoder)"""
        inpt_x = tf.concat([self.qc, self.qa], 1)
        hx = self.fc_net(inpt_x, ((args.nh - 1) * [args.h]), [], 'px_ca_shared', lamba=(args.lamba), activation=activation)
        x_1_hat = self.fc_net(hx, [args.h], [[5, None]], 'px_ca_1', lamba=(args.lamba), activation=activation)
        x_2_hat = self.fc_net(hx, [args.h], [[4, None]], 'px_ca_2', lamba=(args.lamba), activation=activation)
        inpt_t = tf.concat([self.qz, self.qc, self.qu], 1)
        t_hat = self.fc_net(inpt_t, ((args.nh - 1) * [args.h]), [[1, None]], 'pt_zcu', lamba=(args.lamba), activation=activation)
        inpt_y = tf.concat([self.qa, self.qc,self.qu, self.t], 1)
        y_hat = self.fc_net(inpt_y, ((args.nh - 1) * [args.h]), [[1, None]], 'py_caut', lamba=(args.lamba), activation=activation)
        x_1_recon = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=(self.x_1_one_hot), logits=x_1_hat))
        x_2_recon = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=(self.x_2_one_hot), logits=x_2_hat))
        y_recon = tf.reduce_mean(tf.keras.losses.MSE(self.y, y_hat))
        self.y_loss = y_recon
        t_recon = tf.reduce_mean(tf.keras.losses.MSE(self.t, t_hat))
        self.recon_loss = args.loss_y * y_recon + args.loss_x * (x_1_recon + x_2_recon) + args.loss_t * t_recon
        sigmaq_a = self.diagonal(sigmaq_a)
        sigmaq_c = self.diagonal(sigmaq_c)
        sigmaq_z = self.diagonal(sigmaq_z)
        sigmaq_u = self.diagonal(sigmaq_u)
        muq = tf.concat([muq_z, muq_c, muq_a, muq_u], 1)
        sigmaq = tf.concat([sigmaq_z, sigmaq_c, sigmaq_a, sigmaq_u], 1)
        u_dist = tfd.MultivariateNormalDiag(loc=(tf.layers.flatten(muq)), scale_diag=(tf.exp(tf.layers.flatten(sigmaq))))
        u_prior = tfd.MultivariateNormalDiag(loc=(tf.zeros(tf.shape(muq))), scale_diag=(tf.ones(tf.shape(sigmaq))))
        self.kl_loss = args.kl_loss * self.wassertein_distance(muq,sigmaq)
        zcau = tf.concat([self.qz, self.qc, self.qa, self.qu], 1)
        zcau_perm = self.permute_dims(zcau, args.d)
        zcau_total = tf.concat([zcau, zcau_perm], 0)
        D_zcau_total = tf.squeeze(self.fc_net(zcau_total, ((args.nh - 1) * [args.h]), [[2, None]], 'discriminator', lamba=(args.lamba), activation=activation))
        D_zcau, D_zcau_perm = tf.split(D_zcau_total, [tf.shape(zcau)[0], tf.shape(zcau)[0]], 0)
        zeros = tf.zeros(tf.shape(zcau)[0], tf.int32)
        ones = tf.ones(tf.shape(zcau)[0], tf.int32)
        zero_one_hot = tf.one_hot(indices=zeros, depth=2, axis=(-1))
        one_one_hot = tf.one_hot(indices=ones, depth=2, axis=(-1))
        labels_total = tf.concat([zero_one_hot, one_one_hot], 0)
        self.discriminator_loss = 0.5 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=zero_one_hot, logits=D_zcau)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_one_hot, logits=D_zcau_perm))
        self.vae_ad_loss = args.ad_loss * tf.reduce_mean(D_zcau[:, :1] - D_zcau[:, 1:])
        self.vae_loss = self.recon_loss + self.kl_loss + self.vae_ad_loss


def generate_IV(data, resultDir, exp, input_args, iv_dir):
    try:
        args = get_FLAGS()
    except:
        args = tf.app.flags.FLAGS

    seed = args.seed
    args.loss_y = input_args.loss_y
    args.loss_x = input_args.loss_x
    args.loss_t = input_args.loss_t
    args.kl_loss = input_args.kl_loss
    args.ad_loss = input_args.ad_loss
    modelDir = iv_dir + 'models/'
    os.makedirs((os.path.dirname(modelDir)), exist_ok=True)
    logfile = f"{iv_dir}log_parameters.txt"
    modelfile = f"{modelDir}m6-best"
    if args.rewrite_log:
        f = open(logfile, 'w')
        f.close()
    M = None

    s3 = '\nReplication {}/{}'.format(exp + 1, args.reps)
    log(logfile, s3)
    train = data.train
    val = data.valid
    test = data.test
    try:
        train.to_numpy()
        val.to_numpy()
        test.to_numpy()
    except:
        pass

    train = {'x':train.x, 
     't':train.t, 
     'e':train.e, 
     'y':train.y, }
    val = {'x':val.x, 
     't':val.t, 
     'e':val.e, 
     'y':val.y, }
    test = {'x':test.x, 
     'e':test.e, 
     't':test.t, 
     'y':test.g, }
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()
        np.random.seed(seed)
        tf.set_random_seed(seed)
        x_ph_1 = tf.placeholder((tf.int32), [M], name='x_1')
        x_ph_2 = tf.placeholder((tf.int32), [M], name='x_2')
        t_ph = tf.placeholder((tf.float32), [M, 1], name='t')
        y_ph = tf.placeholder((tf.float32), [M, 1], name='y')
        q_labels = tf.placeholder((tf.bool), name='q_labels')
        model = VIV(x_ph_1, x_ph_2, y_ph, t_ph, args, q_labels)
        z = trainNet(model, sess, train, val, test, args, logfile, modelfile, exp)
        np.savez((iv_dir + f"z_{exp}.npz"), rep_z=z)
        data.train.z = z
        sess.close()


def trainNet(model, sess, train, val, test, args, logfile, modelfile, exp):
    best_vae = np.inf
    best_intervention = np.inf
    dict_train = {model.x_1: train['x'][:, 0], model.x_2: train['x'][:, 1], model.y: train['y'], model.t: train['t'], model.q_labels: 0}
    dict_valid = {model.x_1: val['x'][:, 0], model.x_2: val['x'][:, 1], model.y: val['y'], model.t: val['t'], model.q_labels: 0}
    dict_test = {model.x_1: test['x'][:, 0], model.x_2: test['x'][:, 1], model.y: test['y'], model.t: test['t'], model.q_labels: 0}
    max_step = tf.compat.v1.Variable(0, trainable=False, name='max_step')
    min_step = tf.compat.v1.Variable(0, trainable=False, name='min_step')
    max_lr = tf.compat.v1.train.exponential_decay((args.lrate), max_step, (args.lrate_decay_num), (args.lrate_decay), staircase=True)
    min_lr = tf.compat.v1.train.exponential_decay((args.lrate_min), min_step, (args.lrate_decay_num), (args.lrate_decay), staircase=True)
    max_opt = None
    min_opt = None
    if args.optimizer == 'Adagrad':
        max_opt = tf.train.AdagradOptimizer(max_lr)
        min_opt = tf.train.AdagradOptimizer(min_lr)
    else:
        if args.optimizer == 'GradientDescent':
            max_opt = tf.train.GradientDescentOptimizer(max_lr)
            min_opt = tf.train.GradientDescentOptimizer(min_lr)
        else:
            if args.optimizer == 'Adam':
                max_opt = tf.compat.v1.train.AdamOptimizer(max_lr)
                min_opt = tf.compat.v1.train.AdamOptimizer(min_lr)
            else:
                max_opt = tf.compat.v1.train.RMSPropOptimizer(max_lr, args.decay)
                min_opt = tf.compat.v1.train.RMSPropOptimizer(min_lr, args.decay)
    saver = tf.train.Saver(tf.contrib.slim.get_variables())
    variable_names = [variable.name for variable in tf.global_variables()]
    params_max = []
    params_min = []
    for scope_name in variable_names:
        if 'discriminator' in scope_name:
            params_max = params_max + tf.compat.v1.get_collection((tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES), scope=scope_name)
        else:
            params_min = params_min + tf.compat.v1.get_collection((tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES), scope=scope_name)

    grads, global_norm = tf.clip_by_global_norm(tf.gradients(model.vae_loss, params_min), 0.1)
    train_min = min_opt.apply_gradients(zip(grads,params_min),global_step=min_step)
    train_max = max_opt.minimize((model.discriminator_loss), global_step=max_step, var_list=params_max)
    sess.run(tf.global_variables_initializer())
    objnan = False
    n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(train['x'].shape[0] / 100), np.arange(train['x'].shape[0])
    for epoch in range(n_epoch):
        t0 = time.time()
        widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
        pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
        pbar.start()
        np.random.shuffle(idx)
        for j in range(n_iter_per_epoch):
            pbar.update(j)
            batch = np.random.choice(idx, args.bs)
            x_batch, y_batch, t_batch = train['x'][batch, :], train['y'][batch], train['t'][batch]
            dict_batch = {model.x_1: x_batch[:, 0], model.x_2: x_batch[:, 1], model.y: y_batch, model.t: t_batch, model.q_labels: 0}
            if not objnan:
                sess.run(train_max, feed_dict=dict_batch)
                sess.run(train_min, feed_dict=dict_batch)
                if j % args.output_delay == 0 or j == n_iter_per_epoch - 1:
                    vae_loss, recon_loss, kl_loss, ad_loss, disc_loss = sess.run([model.vae_loss,
                     model.recon_loss, model.kl_loss, model.vae_ad_loss, model.discriminator_loss],
                      feed_dict=dict_train)
                    vae_loss_valid, recon_loss_valid, kl_loss_valid, ad_loss_valid = sess.run([model.vae_loss,
                     model.recon_loss, model.kl_loss, model.vae_ad_loss],
                      feed_dict=dict_valid)
                    vae_loss_test, recon_loss_test, kl_loss_test, ad_loss_test = sess.run([model.vae_loss,
                     model.recon_loss, model.kl_loss, model.vae_ad_loss],
                      feed_dict=dict_test)
                    if np.isnan(vae_loss):
                        log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % exp)
                        objnan = True
                        break
                    loss_str = str(epoch) + '_' + str(j) + '_train:' + '\tVaeloss: %.3f,\trecon_loss: %.3f,\tkl_loss: %.3f,\tad_loss: %.3f,\tdiscriminator_loss%.3f' % (
                     vae_loss, recon_loss, kl_loss, ad_loss, disc_loss)
                    loss_str_valid = str(epoch) + '_' + str(j) + '_valid:' + '\tVaeloss: %.3f,\trecon_loss: %.3f,\tkl_loss: %.3f,\tad_loss: %.3f, ' % (
                     vae_loss_valid, recon_loss_valid, kl_loss_valid, ad_loss_valid)
                    loss_str_test = str(epoch) + '_' + str(j) + '_test:' + '\tVaeloss: %.3f,\trecon_loss: %.3f,\tkl_loss: %.3f,\tad_loss: %.3f ' % (
                     vae_loss_test, recon_loss_test, kl_loss_test, ad_loss_test)

                    log(logfile, loss_str)
                    log(logfile, loss_str_valid)
                    log(logfile, loss_str_test)

                    vae_valid = sess.run((model.vae_loss), feed_dict=dict_valid)
                    if vae_valid <= best_vae:
                        saver.save(sess, modelfile)
                        str_loss = 'Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_vae, vae_valid)
                        best_vae = vae_valid
                        log(logfile, str_loss)

        if objnan:
            break
        if epoch % args.earl == 0 or epoch == n_epoch - 1:
            vae_valid = sess.run((model.vae_loss), feed_dict=dict_valid)
            if vae_valid <= best_vae:
                saver.save(sess, modelfile)
                str_loss = 'Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_vae, vae_valid)
                best_vae = vae_valid
                log(logfile, str_loss)

    saver.restore(sess, modelfile)
    z = sess.run((model.qz), feed_dict=dict_train)
    return z


def get_IV(data, exp, iv_dir):
    load_z = np.load(iv_dir + f"z_{exp}.npz")
    data.train.z = load_z['rep_z']