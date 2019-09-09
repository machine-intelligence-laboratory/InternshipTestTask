import pymorphy2


class Pymorphy2Tagger():
    """
    Pymorphy2-based tagger. Per-token tagging is used.
    Works only for russian language.

    You need to install pymorphy2 package to use this class.
    """
    def __init__(self):
        self._pymorphy2_engine = pymorphy2.MorphAnalyzer()
        
        self._opencorpora_ners = {
            'Name',  # имя
            'Surn',  # фамилия
            'Patr',  # отчество
            'Geox',  # гео-тег
            'Orgn',  # организация
            'Trad',  # торговая марка
            'Abbr',  # аббревиатура
        }

        self._map_to_PLO_notation = {
            'Name': 'PER',
            'Surn': 'PER',
            'Patr': 'PER',
            'Geox': 'LOC',
            'Orgn': 'ORG',
            'Trad': 'ORG',
            'Abbr': 'ORG',
            None: None
        }

    def transform_token(self, token):
        """
        Apply tagger to one token.

        Parameters
        ----------
        token : str
            One token. Supposed to be a single word.

        Returns
        -------
        transformed_token : TaggerNormalizerResult
            Contains all main extracting results.
        """
        assert isinstance(token, str), 'token must have str type'

        transformed_token = self._pymorphy2_engine.parse(token)

        # None if tag.grammemes does not contain ner tags
        token_raw_ner_tag = self._opencorpora_ners.intersection(transformed_token[0].tag.grammemes)
        # length of taken_raw_ner_tag is 0 or 1
        token_raw_ner_tag = list(token_raw_ner_tag)[0] if token_raw_ner_tag else None

        # parse all default fields
        transformed_token = {
            'initial_form': transformed_token[0].word,
            'normal_form': transformed_token[0].normal_form,
            'pos_tag': transformed_token[0].tag.POS,
            'ner_tag': self._map_to_PLO_notation[token_raw_ner_tag]
        }

        return transformed_token

    def transform_string(self, tokenized_string):
        """
        Apply tagger to one tokenized string transformation.

        Parameters
        ----------
        tokenized_string : iterable of str
            List of tokens (iterable of tokens can be work too).
            Supposed to be one sentence or one document.

        Returns
        -------
        transformed_string : list of TaggerNormalizerResult
            List of namedtuples. Each namedtuple is structured result of extraction.
        """
        if isinstance(tokenized_string, str):
            raise TypeError("tokenized_string can't be str, use transform_token for single str")

        transformed_string = [
            self.transform_token(token)
            for token in tokenized_string
        ]

        return transformed_string

    def transform(self, tokenized_collection):
        """
        Apply tagger to collection of tokenized strings or documents.

        Parameters
        ----------
        tokenized_collection : iterable of iterable of str
            Collection of tokenized sentences or documents to be transformed.

        Returns
        -------
        transformed_collection : list of optional
            Collection of transformed objects.
        """
        transformed_collection = [
            self.transform_string(element)
            for element in tokenized_collection
        ]
        return transformed_collection

    def parse(self, token):
        """
        Raw pymorphy2 parsing for one token.

        Parameters
        ----------
        token : str
            One token. Supposed to be a single word.

        Returns
        -------
        transformed_token : list of TaggerNormalizerResult
            Contains all extracting results!
        """
        assert isinstance(token, str), 'token must have str type'

        transformed_token = self._pymorphy2_engine.parse(token)

        return transformed_token
