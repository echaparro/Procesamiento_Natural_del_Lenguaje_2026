import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer,LancasterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string 
import unicodedata
import pandas as pd
import re
from typing import Any, Set, Optional, List, Tuple
import spacy
from collections import Counter
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp_es = spacy.load("es_core_news_sm")

class NLPTextCleaning:

    _URL_PATTERN = re.compile(r'https?://[^\s]+\.?[^\s]*')
    _EMAIL_PATTERN = re.compile(r'[\w.]+@[\w.]+\.[a-zA-Z]{2,}')
    _MENTION_PATTERN = re.compile(r'@\w+')
    _HASHTAG_PATTERN = re.compile(r'#\w+')
    _NUMBER_PATTERN = re.compile(r'\S*\d\S*')
    _EMOJI_PATTERN = re.compile(r'[^\x00-\x7Fáéíóúüñ¡¿ÁÉÍÓÚÜÑa-zA-Z \n.,!?]')
    _PUNCTUATION_PATTERN = re.compile(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', re.UNICODE)
    _SPACE_PATTERN = re.compile(r'\s+')

    DEFAULT_OPTIONS = {"urls", "correos", "menciones", "hashtags",
                      "numeros", "emojis", "puntuacion", "espacios"}
    DEFAULT_OPTIONS_STEAM = {"Porter", "Snowball", "Lancaster"}
    
    def __init__(self, 
                 idioma: str = 'spanish',
                 opciones_normalizar: Set[str] = None,#{"urls", "correos", "menciones", "hashtags", "numeros", "emojis", "puntuacion", "espacios"},
                 remover_acentos: bool = True,
                 remover_stopwords: bool = True,
                 normalizar_unicode: bool = True,
                 metodo_stem: Optional[str] = None):
        
        self.idioma = idioma
        self.stopswords = set(stopwords.words('spanish' if idioma=='spanish' else 'english'))
        self.lematizer = WordNetLemmatizer()
        self.metodo_stem = metodo_stem
        
        self.opciones_a_normalizar = opciones_normalizar or self.DEFAULT_OPTIONS

        self.remover_acentos = remover_acentos
        self.remover_stopwords = remover_stopwords
        self.normalizar_unicode = normalizar_unicode

        self._operaciones = {
            'urls': self._limpiar_urls,
            'correos': self._limpiar_emails,
            'menciones': self._limpiar_menciones,
            'hashtags': self._limpiar_hashtags,
            'numeros': self._limpiar_numeros,
            'emojis': self._limpiar_emojis,
            'puntuacion': self._limpiar_puntuacion,
            'espacios': self._normalizar_espacios
        }
    
    #------------Funciones de limpieza---------------------------------------------------------
    def _limpiar_urls(self, parTexto: str) -> str:
        return self._URL_PATTERN.sub('', parTexto)
    
    def _limpiar_emails(self, parTexto: str) -> str:
        return self._EMAIL_PATTERN.sub('', parTexto)
    
    def _limpiar_menciones(self, parTexto: str) -> str:
        return self._MENTION_PATTERN.sub('', parTexto)
    
    def _limpiar_hashtags(self, parTexto: str) -> str:
        return self._HASHTAG_PATTERN.sub('', parTexto)
    
    def _limpiar_numeros(self, parTexto: str) -> str:
        return self._NUMBER_PATTERN.sub('', parTexto)
    
    def _limpiar_emojis(self, parTexto: str) -> str:
        return self._EMOJI_PATTERN.sub('', parTexto)
    
    def _limpiar_puntuacion(self, parTexto: str) -> str:
        return self._PUNCTUATION_PATTERN.sub(' ', parTexto)
    
    def _normalizar_espacios(self, parTexto: str) -> str:
        return self._SPACE_PATTERN.sub(' ', parTexto).strip()
    
    
    #------------Tokenizacion---------------------------------------------------------
    def Tokenizar_Regex(self, parTexto: str) -> list:
        return re.findall(r"\b\w+\b", parTexto, re.UNICODE)

    def Tokenizar_Oraciones_NLTK(self, parTexto: str) -> list:
        return sent_tokenize(parTexto, language=self.idioma)
    
    def Tokenizar_Oraciones(self, parTexto: str) -> list:
        """Divide texto en oraciones usando regex."""
        patron = r'(?<=[.!?¿¡])\s+(?=[A-ZÁÉÍÓÚ])|(?<=[.!?])\s*$'
        oraciones = re.split(patron, parTexto.strip())
        return [s.strip() for s in oraciones if s.strip()]

    def Tokenizar_Palabras_NLTK(self, parTexto: str) -> list:
        return word_tokenize(parTexto, language=self.idioma)
    
    def tokenizar_oraciones(self, parTexto:str):
        texto = parTexto.lower()
        oraciones = sent_tokenize(texto)       
        oraciones_tokenizadas = [] 
        for oracion in oraciones:
            tokens = word_tokenize(oracion)
            tokens_limpios = []
            for token in tokens:
                if token.isalpha() and len(token) > 1:
                    tokens_limpios.append(token)
            
            if tokens_limpios:
                oraciones_tokenizadas.append(tokens_limpios)
        
        return oraciones_tokenizadas

#------------Stemming y Lematizacion---------------------------------------------------------
    def stemmeing(self, parTexto: str) :
        para_stemming = self.preprocesar(parTexto)
        stemMetodo = None
         # Español: forzar SnowballStemmer, las otras dos no soportan español
        if self.idioma.lower() == 'spanish':
            stemMetodo = SnowballStemmer('spanish')
        
        # Ingles
        elif self.idioma.lower() == 'english':
            match self.metodo_stem:
                case "Porter":
                    stemMetodo = PorterStemmer()
                case "Snowball":
                    stemMetodo = SnowballStemmer(self.idioma)
                case "Lancaster":
                    stemMetodo = LancasterStemmer()
                case _:
                    stemMetodo = SnowballStemmer(self.idioma) # default a Snowball si no se especifica o no se reconoce el método
        
        else:
            stemMetodo = SnowballStemmer('english') #default a Snowball para otros idiomas si están soportados
        
        if stemMetodo:
            tokens = self.Tokenizar_Palabras_NLTK(para_stemming)
            return [stemMetodo.stem(token) for token in tokens]
        else: return None

    def graficar_stemming(self, vocabularios: list[dict], parTitulo:str):
        metodos = [item['Metodo'] for item in vocabularios]
        zise = [item['Vocabulario'] for item in vocabularios]
        colores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A9F6A']
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(metodos, zise, color=colores[:len(metodos)], width=0.6)
        ax.bar_label(bars, fontsize=11, fontweight='bold')
        if 'Original' in metodos:
            original_idx = metodos.index('Original')
            original = zise[original_idx]
            for i, metodo in enumerate(metodos):
                if metodo != 'Original':
                    reduccion = ((original - zise[i]) / original) * 100
                    ax.text(i, zise[i] + 1, f'-{reduccion:.1f}%', 
                        ha='center', fontsize=10, fontweight='bold')
    
        ax.set_title(parTitulo, fontsize=13, fontweight='bold')
        ax.set_ylabel('Número de palabras únicas', fontsize=11)
        ax.set_xlabel('Método', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()




    def lematizacion(self, parTexto: str):
        para_lematizacion = self.preprocesar(parTexto)
        lemmatizer  = None
         # Español: para español se puede usar WordNetLemmatizer pero no es tan
        if self.idioma.lower() == 'english':
            lemmatizer  = WordNetLemmatizer()
        elif self.idioma.lower() == 'spanish':
            doc = nlp_es(para_lematizacion)
            lemas = []
            for token in doc:
                if not token.is_punct and not token.is_space:
                    lemas.append(token.lemma_)
        
            return lemas
                    
        if lemmatizer :
            tokens = self.Tokenizar_Palabras_NLTK(para_lematizacion)
            return [lemmatizer .lemmatize(token) for token in tokens]
        else: return None


    def normalizar(self, parTexto: str) -> str:
        resultado = parTexto
        orden_operaciones = ['urls', 'correos', 'menciones', 'hashtags', 
                         'numeros', 'emojis', 'puntuacion', 'espacios']
        for opcion in orden_operaciones:
            if opcion in self.opciones_a_normalizar:
                match opcion:
                    case 'urls':
                        resultado = self._limpiar_urls(resultado)
                    case 'correos':
                        resultado = self._limpiar_emails(resultado)
                    case 'menciones':
                        resultado = self._limpiar_menciones(resultado)
                    case 'hashtags':
                        resultado = self._limpiar_hashtags(resultado)
                    case 'numeros':
                        resultado = self._limpiar_numeros(resultado)
                    case 'emojis':
                        resultado = self._limpiar_emojis(resultado)
                    case 'puntuacion':
                        resultado = self._limpiar_puntuacion(resultado)
                    case 'espacios':
                        resultado = self._normalizar_espacios(resultado)
                    case _:
                        pass
        
        return resultado.lower()

    def remover_acentosM(self, parTexto: str) -> str:
        """Elimina tildes y diacríticos usando normalización Unicode."""

        if not self.remover_acentos:
         return parTexto
    
        nfkd = unicodedata.normalize('NFKD', parTexto)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    def remover_stopwordsM(self, parText:str) -> str:
        """Elimina stopwords de una lista de tokens."""
        if not self.remover_stopwords:
                    return parText      

        tokens = self.get_vocabulario(parText)
        if not self.remover_stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stopswords]
    
    def graficar_frecuencia(self, freq_con: List[tuple[Any, int]], freq_sin: List[tuple[Any, int]], parTitulo: str, top_n: int = 10):
        """
        Grafica la frecuencia de palabras con y sin stopwords
        
        Args:
            freq_con: Lista de tuplas (palabra, frecuencia) CON stopwords
            freq_sin: Lista de tuplas (palabra, frecuencia) SIN stopwords
            parTitulo: Título del gráfico
            top_n: Número de palabras más frecuentes a mostrar
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        sin_stopwords_freq = Counter(freq_sin).most_common(top_n)
        con_stopwords_freq = Counter(freq_con).most_common(top_n)
        for ax, data, titulo, color in [
            (axes[0], con_stopwords_freq, 'Con stopwords', 'slategray'),
            (axes[1], sin_stopwords_freq, 'Sin stopwords', 'steelblue')
        ]:
            words, counts = zip(*data)
            ax.barh(list(reversed(words)), list(reversed(counts)), color=color)
            ax.set_title(titulo, fontsize=11)
            ax.set_xlabel('Frecuencia')
        
        plt.suptitle(parTitulo, fontsize=12)
        plt.tight_layout()
        plt.show()
    

    def get_vocabulario(self, parTexto: str) -> Set[str]:
        """Devuelve el vocabulario único de un texto después de preprocesar."""
        texto_preprocesado = self.preprocesar(parTexto)
        tokens = self.Tokenizar_Palabras_NLTK(texto_preprocesado)
        return set(tokens)
    
    def preprocesar(self, parTexto: str) -> str:
       if not parTexto:
            return ""
       #1.- Normalizacion
       texto_normalizado = self.normalizar(parTexto)
       #2.- Quitar acentos
       texto_normalizado=self.remover_acentosM(texto_normalizado)

       return texto_normalizado

