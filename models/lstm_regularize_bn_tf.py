from datetime import datetime

import tensorflow as tf

runtime = datetime.now().strftime('%d_%m_%H_%M')
summaries_dir = 'logs/tf_lstm_%s/' % runtime
weight_path = 'logs/tf_lstm_%s/model' % runtime


def _lstm_cell(layer_shape, prob):
    drop_out = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_unit), output_keep_prob=prob) for num_unit in layer_shape]
    return drop_out


def train(x_train, y_train, x_test, y_test, layer_shape, time_steps, epoch, learning_rate, predict_length):
    num_feature = x_train[0].shape[-1]
    print(num_feature)
    tf.reset_default_graph()
    tf.set_random_seed(1)
    inputs = tf.placeholder(tf.float32, [None, time_steps, num_feature], name='input')
    targets = tf.placeholder(tf.float32, [None, predict_length], name='targets')
    # multiLSTMCell = tf.nn.rnn_cell.MultiRNNCell(_lstm_cell(layer_shape))
    prob = tf.placeholder_with_default(1.0, shape=())
    multiLSTMCell = tf.nn.rnn_cell.MultiRNNCell(_lstm_cell(layer_shape, prob))
    value_rnn, state = tf.nn.dynamic_rnn(multiLSTMCell, inputs=inputs, dtype=tf.float32)
    w = tf.get_variable(name='w_ho', shape=(layer_shape[-1], predict_length),
                        initializer=tf.initializers.truncated_normal())
    bias = tf.get_variable(name='b_ho', shape=predict_length,
                           initializer=tf.initializers.constant((1, layer_shape[-1])))
    tf.summary.histogram('w', w)
    tf.summary.histogram('b', bias)
    val = tf.transpose(value_rnn, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    output = tf.nn.xw_plus_b(last, w, bias, name='output')
    loss = tf.losses.mean_squared_error(targets, output)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='eve').minimize(loss)
    saver = tf.train.Saver()
    accuracy = tf.reduce_mean(loss)

    train_accuracy = tf.summary.scalar('train_accuracy', accuracy)
    validation_accuracy = tf.summary.scalar("validation_accuracy", accuracy)

    init = tf.global_variables_initializer()
    summ = tf.summary.merge_all()
    # batch size tool tf
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
        # train
        for epoch_step in range(epoch):
            # sess.run((p, optimizer), feed_dict={inputs: x_train, targets: y_train})
            sess.run(optimizer, feed_dict={inputs: x_train, targets: y_train, prob: 0.5})
            # evaluation
            if epoch_step % 5 == 0:
                print('Step %d/%d' % (epoch_step, epoch))
                [train_acc, s] = sess.run([train_accuracy, summ], feed_dict={inputs: x_train, targets: y_train, prob: 1.0})
                train_writer.add_summary(s, global_step=epoch_step)
                val_acc = sess.run(validation_accuracy, feed_dict={inputs: x_test, targets: y_test, prob: 1.0})
                train_writer.add_summary(train_acc, global_step=epoch_step)
                train_writer.add_summary(val_acc, global_step=epoch_step)
        train_writer.close()
        saver.save(sess, weight_path)
