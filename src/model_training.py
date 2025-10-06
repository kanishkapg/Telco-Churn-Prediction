import os
import joblib

class ModelTrainer:
    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        return model, train_score

    def save_model(self, model, filepath):
        joblib.dump(model, filepath)
        
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        return self.model
