from ml_models.destination_classifier import train_model, predict_category

train_model()

print(predict_category("warm", "Paris", "medium")) 
