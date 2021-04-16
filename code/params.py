class params():
    def __init__(self):
        self.source_dir = "../.."
        self.batch_size = 4
        self.learning_rate =  1e-7
        self.max_len = 128
        self.num_epochs = 3
        self.random_state = 9999
        self.num_folds = 5
        self.task = "single"
        self.batch_weight = None
        self.sort_by = None
        self.stratified = True
        self.predict = "label"
        self.mc_passes = 10

    def update(self, new):
        for k, v in new.__dict__.items():
            if getattr(new, k) is not None:
                print("Changing the default value of {} from {} to {}".format(k, getattr(self, k), v))
                setattr(self, k, v)

    def update_dict(self, new):
        for k, v in new.items():
              print("Changing the default value of {} from {} to {}".format(k, getattr(self, k), v))
              setattr(self, k, v)
