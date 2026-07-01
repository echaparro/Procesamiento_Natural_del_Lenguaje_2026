import os
import torch
from enum import Enum
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODELOS_DIR = os.path.join(os.path.dirname(__file__), "Modelos")

class ModelName(Enum):
    DISTILBERT ='DistilBERT'
    #BETO ='BETO'
    #ROBERTA='ROBERTA'
    BERTIN='BERTIN'

class chat:
    def __init__(self, parModelo: ModelName = ModelName.DISTILBERT):
        if isinstance(parModelo, str):
            parModelo = ModelName(parModelo)

        match parModelo:
            #case ModelName.BETO:
             #   self.hf_model_name = 'MMG/xlm-roberta-large-squad2-es'
            case ModelName.DISTILBERT:
                self.hf_model_name = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
            #case ModelName.ROBERTA:
             #   self.hf_model_name = 'xlm-roberta-large-finetuned-squad2-es'
            case ModelName.BERTIN:
                self.hf_model_name = 'mrm8488/longformer-base-4096-spanish-finetuned-squad'

        local_dir = os.path.join(MODELOS_DIR, parModelo.value)
        if os.path.exists(local_dir):
            self.model = AutoModelForQuestionAnswering.from_pretrained(local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.hf_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            os.makedirs(local_dir, exist_ok=True)
            self.model.save_pretrained(local_dir)
            self.tokenizer.save_pretrained(local_dir)
        self.model_choice = parModelo

    def get_model_name(self):
        return self.model_choice.value
    
    def answer_question(self, parQuestion, parContext):
        if not parQuestion or not parContext:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        tokenizer = self.tokenizer
        model = self.model
        match self.model_choice:
            case ModelName.DISTILBERT: #| ModelName.BETO |  ModelName.ROBERTA:
                max_len = 512  # Máximo soportado
            case ModelName.BERTIN:
                max_len = 4096

        inputs = tokenizer(
            parQuestion,
            parContext,
            return_tensors="pt",
            truncation="only_second",
            max_length=max_len,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        start_scores = torch.softmax(outputs.start_logits, dim=1)
        end_scores = torch.softmax(outputs.end_logits, dim=1)

        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        if end_idx < start_idx:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        confidence = (start_scores[0][start_idx] * end_scores[0][end_idx]).item()

        return {
            "answer": answer,
            "score": confidence,
            "start": start_idx.item(),
            "end": end_idx.item(),
            "model":self.model_choice.value
        }
