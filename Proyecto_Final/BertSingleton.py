from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class BertSingleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            # Aquí va la inicialización real
            self.model_name = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
            # Cargar modelo y tokenizador
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def get_tokeniker(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model