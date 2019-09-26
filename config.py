class Config(object):
    # general setting
    input_dir = "./sample_data/"
    output_dir = "./run/"
    
    # model setting
    input_channel = 3
    output_channel = 5
    hidden_channel = 128
    hidden_layer = 3
    
    # training setting
    batch_size = 5
    epoch = 2
    