from src.models.baseline import BaselineCNN

def get_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model'].get('pretrained', False)

    if model_name == 'baseline':
        return BaselineCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")