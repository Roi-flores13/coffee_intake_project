def define_search_space(trial, model_name):
    """
    Devuelve un diccionario de parámetros basado en el 'trial' de Optuna.
    """
    params = {} # Dictionary to save all parameters
    
    if model_name == 'LogisticRegression':
        params['C'] = trial.suggest_float('C', 1e-2, 1e2, log=True) # Regularization
        params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2']) # Regularization
        
        # Fixed parameters
        params['solver'] = 'saga'
        params['max_iter'] = 1000
        
    elif model_name == 'RandomForest':
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500, step=25) # Parameter n_estimators
        params['max_depth'] = trial.suggest_int('max_depth', 3, 30) # Parameter max_depth
        params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) # Parameter max_features
        params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy']) # Parameter criterion
        
    elif model_name == 'MLP':
        params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes', 
            [(50,), (100,), (50, 50), (100, 50), (100, 100, 50)]    # Number of layers and neruons
        )
        params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh']) # Activation function
        params['alpha'] = trial.suggest_float('alpha', 1e-4, 1e-1, log=True) # Regularization
        params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 1e-3, 1e-1, log=True) # Initial learning rate
        
        #Fixed parameters
        params['solver'] = 'adam'
        params['max_iter'] = 500

    return params