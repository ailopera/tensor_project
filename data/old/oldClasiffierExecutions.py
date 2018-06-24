# configuraciones antiguas
# iterations = [
        # { "activation_function": "relu", "config_tag": "original", "hidden_neurons": [300, 100]}, #Configuracion original
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [400, 200]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [500, 300]},
    
        #{ "activation_function": "relu", "config_tag": "epochs", "hidden_neurons": [500, 300], "epochs": 25},    
        #{ "activation_function": "relu", "config_tag": "epochs", "hidden_neurons": [500, 300], "epochs": 30},
        #{ "activation_function": "relu", "config_tag": "epochs", "hidden_neurons": [500, 300], "epochs": 35},
        
        #{ "activation_function": "relu", "config_tag": "activation", "hidden_neurons": [500, 300]},
        #{ "activation_function": "leaky_relu", "config_tag": "activation", "hidden_neurons": [500, 300]},
        #{ "activation_function": "elu", "config_tag": "activation", "hidden_neurons": [500, 300]},
        
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [550, 350]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [400, 400]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [300, 300]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [200, 100]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [200, 50]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [50, 25]},

        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 225, 150]}, # 3 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [500, 275, 200]}, # 3 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 300, 225, 150]}, # 4 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [375, 225, 150, 75]}, # 4 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 375, 300, 225, 150]}, # 5 capas
        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [375,300, 225, 150, 75]},  # 5 capas

        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 225, 150], "learning_rate": 0.1}, # 3 capas
        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [375, 225, 150, 75], "learning_rate": 0.1}, # 4 capas
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [375,300, 225, 150, 75], "learning_rate": 0.1} # 5 capas

        #{ "activation_function": "relu", "config_tag": "dropout", "hidden_neurons": [300, 100], "dropout_rate": 0.25},
        #{ "activation_function": "relu", "config_tag": "dropout", "hidden_neurons": [300, 100], "dropout_rate": 0.50},
        #{ "activation_function": "relu", "config_tag": "dropout", "hidden_neurons": [300, 100], "dropout_rate": 0.75}

        # { "activation_function": "relu", "config_tag": "architecture_shallow", "hidden_neurons": [500, 250],"learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [100, 100, 100, 100], "learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [150, 125, 100, 75], "learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [125, 100, 100, 75, 50],"learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [100, 75, 50, 50, 25], "learning_decrease": True, "learning_rate": 0.1} 
        
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [300, 100],"dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [500, 250],"dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [250, 100, 50], "dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [150, 125, 100, 75], "dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [125, 100, 100, 75, 50], "dropout_rate": 0.50}, 

        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [300, 100],"dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [500, 250],"dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [250, 100, 50], "dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [150, 125, 100, 75], "dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [125, 100, 100, 75, 50], "dropout_rate": 0.75},
        
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [500, 250], "learning_decrease": 0.9, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [100, 100, 100, 100], "learning_decrease": 0.9, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [150, 125, 100, 75], "learning_decrease": 0.9, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [125, 100, 100, 75, 50],"learning_decrease": 0.9, "learning_rate": 0.1}
        #{ "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [100, 75, 50, 50, 25], "learning_decrease": 0.9, "learning_rate": 0.1}          # Ejecuciones aumentando el tamano del batch
        #{ "activation_function": "relu", "config_tag": "ampliando_batch", "hidden_neurons": [300, 100], 'batch_size': 150}  
# ]