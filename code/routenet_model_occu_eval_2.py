import tensorflow as tf


class RouteNetModelOccuEval_2(tf.keras.Model):
    """ Init method for the custom model.
    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.
    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(RouteNetModelOccuEval_2, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))

        # Used to mask the input sequence to skip timesteps
        self.masking = tf.keras.layers.Masking()

        # Embedding to compute the initial link hidden state
        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=3),
            tf.keras.layers.Dense(int(int(self.config['HYPERPARAMETERS']['link_state_dim'])),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['link_state_dim']), activation=tf.nn.selu)
        ])

        # Embedding to compute the initial path hidden state
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=3),
            tf.keras.layers.Dense(int(int(self.config['HYPERPARAMETERS']['path_state_dim'])),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['path_state_dim']), activation=tf.nn.selu)
        ])

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=int(self.config['HYPERPARAMETERS']['link_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(output_units)
        ])

    @tf.function
    def call(self, inputs):
        """This function is execution each time the model is called
        Args:
            inputs (dict): Features used to make the predictions.
        Returns:
            tensor: A tensor containing the per-path delay.
        """

        traffic = tf.expand_dims(tf.squeeze(inputs['traffic']), axis=1)
        packets = tf.expand_dims(tf.squeeze(inputs['packets']), axis=1)
        eqlambda = tf.expand_dims(tf.squeeze(inputs['eqlambda']), axis=1)
        link_to_path = tf.squeeze(inputs['link_to_path'])
        path_to_link = tf.squeeze(inputs['path_to_link'])
        path_ids = tf.squeeze(inputs['path_ids'])
        sequence_path = tf.squeeze(inputs['sequence_path'])
        sequence_links = tf.squeeze(inputs['sequence_links'])
        n_links = inputs['n_links']
        n_paths = inputs['n_paths']
        queue_size = tf.expand_dims(tf.squeeze(inputs['qsize']), axis=1)
        capraw = tf.expand_dims(tf.squeeze(inputs['capraw']), axis=1)
        traffraw = tf.expand_dims(tf.squeeze(inputs['traffraw']), axis=1)
        packraw = tf.expand_dims(tf.squeeze(inputs['packraw']), axis=1)

        # Initialize the initial hidden state for paths
        path_state = tf.concat([
            traffic,
            packets,
            eqlambda
        ], axis=1)

        # Initialize the initial hidden state for links
        traffraw_ = tf.concat([
            traffraw
        ], axis=1)

        
        traffic_gather_ = tf.gather(traffraw_, path_to_link)
        traffic_sum_ = tf.math.unsorted_segment_sum(traffic_gather_, sequence_links, n_links)
        link_load = tf.math.divide(traffic_sum_, capraw)
        square_link_load = tf.math.pow(link_load, 2)
        cube_link_load = tf.math.pow(link_load, 3)
        
        link_state = tf.concat([
            link_load,
            square_link_load,
            cube_link_load
        ], axis=1)
        
        #embed the link and path hidden state
        link_state = self.link_embedding(link_state)
        path_state = self.path_embedding(path_state)

        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):
            # The following lines generate a tensor of dimensions [n_paths, max_len_path, dimension_link] with all 0
            # but the link hidden states
            link_gather = tf.gather(link_state, link_to_path)

            ids = tf.stack([path_ids, sequence_path], axis=1)
            max_len = tf.reduce_max(sequence_path) + 1
            shape = tf.stack([
                n_paths,
                max_len,
                int(self.config['HYPERPARAMETERS']['link_state_dim'])])

            # Generate the aforementioned tensor [n_paths, max_len_path, dimension_link]
            link_inputs = tf.scatter_nd(ids, link_gather, shape)

            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)

            path_state_sequence, path_state = path_update_rnn(inputs=self.masking(link_inputs),
                                                              initial_state=path_state)

            # For every link, gather and sum the sequence of hidden states of the paths that contain it
            path_gather = tf.gather(path_state, path_to_link)
            path_sum = tf.math.unsorted_segment_sum(path_gather, sequence_links, n_links)

            # Second message passing: update the link_state
            # The ensure shape is needed for Graph_compatibility
            path_sum = tf.ensure_shape(path_sum, [None, int(self.config['HYPERPARAMETERS']['path_state_dim'])])
            link_state, _ = self.link_update(path_sum, [link_state])

        # Call the readout ANN and return its predictions
        r = self.readout(link_state)

        #apply post-processing to obtain delay estimations from link occupancy
        #capraw is un-modified capacity
        link_delays = tf.divide(tf.multiply(r, queue_size), capraw)
        l_gth = tf.gather(link_delays, link_to_path)
        path_delays = tf.math.unsorted_segment_sum(l_gth, path_ids, n_paths)

        avg_packet_sizes = tf.divide(traffraw, packraw)

        #multiplication by avg packet size
        path_delays = tf.math.multiply(path_delays, avg_packet_sizes)

        return path_delays
