import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import pickle

class EmotionClassifierFromCheckpoint:
    def __init__(self, checkpoint_path, vocab_path=None, class_names=None, device=None):
        """
        Initialize classifier from Lightning checkpoint
        
        Args:
            checkpoint_path: Path to .ckpt file
            vocab_path: Path to saved vocabulary (if separate)
            class_names: List of emotion class names
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load the model from checkpoint
        # Method 1: Load the entire Lightning module
        self.model = MyLSTM.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.freeze()  # Important for inference
        
        # Extract hyperparameters from the model
        self.hparams = self.model.hparams if hasattr(self.model, 'hparams') else {}
        
        # Load vocabulary (you might have saved it separately)
        if vocab_path:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            # Try to get vocab from model or checkpoint
            if hasattr(self.model, 'vocab'):
                self.vocab = self.model.vocab
            else:
                raise ValueError("Vocabulary not found. Provide vocab_path.")
        
        # Get class names
        if class_names:
            self.class_names = class_names
        elif hasattr(self.model, 'class_names'):
            self.class_names = self.model.class_names
        else:
            # If not stored, you'll need to know your classes
            # You might have this from your dataset
            self.class_names = ['happy', 'sad', 'angry', 'peaceful']  # Update with your actual classes
    
    def preprocess_text(self, text, max_length=100):
        """
        Preprocess text to match training format
        """
        # Use the same tokenization as during training
        tokens = text.lower().split()
        
        # Convert to indices
        token_indices = []
        for token in tokens:
            if token in self.vocab.stoi:  # Using torchtext vocab
                token_indices.append(self.vocab.stoi[token])
            else:
                token_indices.append(self.vocab.stoi['<unk>'])  # Use UNK token if available
        
        # Pad/truncate
        if len(token_indices) >= max_length:
            token_indices = token_indices[:max_length]
        else:
            token_indices = token_indices + [0] * (max_length - len(token_indices))
        
        return torch.tensor([token_indices], dtype=torch.long).to(self.device)
    
    def predict(self, text, return_probabilities=False):
        """
        Predict emotion for given lyrics
        """
        with torch.no_grad():
            input_tensor = self.preprocess_text(text)
            outputs = self.model(input_tensor)
            
            # For classification, use softmax
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            if return_probabilities:
                prob_dict = {
                    self.class_names[i]: float(probabilities[0][i])
                    for i in range(len(self.class_names))
                }
                return prob_dict
            else:
                return self.class_names[predicted_class]