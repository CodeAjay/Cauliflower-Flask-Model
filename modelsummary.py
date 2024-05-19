import tensorflow as tf

def generate_model_summary():
    # Load the model
    model = tf.keras.models.load_model('trained_model_m.keras')
    
    # Generate the model summary
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    
    return summary
