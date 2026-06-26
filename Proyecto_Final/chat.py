import torch
from BertSingleton import BertSingleton

class chat:

    def __init__(self, parContext :str, parTitle:str = ""):
        '''Este chat recibe un contexto al cual se le podran hacer preguntas en español '''
        # Cargar modelo y tokenizador
        self.Bert = BertSingleton()
        self.context = parContext
        self.title = parTitle
    
    def get_title(self):
        return self.title

    def answer_question(self, parQuestion):
        # Tokenizar
        inputs = self.Bert.get_tokeniker()(
            parQuestion, 
            self.context,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Obtener predicciones
        with torch.no_grad():
            outputs = self.Bert.get_model()(**inputs)
        
        # Encontrar mejor respuesta
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores) + 1
        
        # Decodificar respuesta
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx]
        answer = self.Bert.get_tokeniker().decode(answer_tokens, skip_special_tokens=True)
        
        # Calcular confianza
        start_score = torch.softmax(start_scores, dim=1)[0][start_idx]
        end_score = torch.softmax(end_scores, dim=1)[0][end_idx-1]
        confidence = (start_score * end_score).item()
        
        return {
            'answer': answer,
            'score': confidence,
            'start': start_idx.item(),
            'end': end_idx.item()
        }
