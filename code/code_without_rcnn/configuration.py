class Configuration():

    def __init__(self, args):
        self.word_threshold = 2
        self.max_len = 20
        self.dim_imgft = 1536
        #self.embedding_size = 256
	self.embedding_size = 64
        #self.num_hidden = 256
	self.num_hidden = 64
        self.batch_size = 40
	#self.batch_size = 64
        self.num_timesteps = 22
        self.learning_rate = 0.002
        self.nb_epochs = 1000
        self.bias_init = True
        self.xavier_init = False
        self.dropout = True
        self.lstm_keep_prob = 0.7
        self.beta_l2 = 0.001
        self.batch_decode = False
        self.mode = args["mode"]
        self.resume = args["resume"]
        self.load_image = args.get("load_image")
        self.saveencoder = bool(args["saveencoder"])
        self.savedecoder = bool(args["savedecoder"])
