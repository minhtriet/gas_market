from keras.callbacks import TensorBoard


class LSTMCallback(TensorBoard):
    def __init__(self, log_dir, max_len):
        super(LSTMCallback, self).__init__(log_dir)
        self.counter = 0
        self.max_len = max_len

    def on_batch_end(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            print('Reset states')
            self.model.reset_states()
        self.counter += 1
