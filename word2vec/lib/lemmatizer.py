from .tagger import Pymorphy2Tagger


class Pymorphy2Lemmatizer():
    """
    Pymorphy2-based lemmatizer. Per-token lemmatization is used.
    Works only for russian language.

    You need to install pymorphy2 package to use this class.
    """
    def __init__(self):
        self._tagger = Pymorphy2Tagger()

    def transform_token(self, token):
        """
        Apply lemmatizer to one token.

        Parameters
        ----------
        token : str
            One token. Supposed to be a single word.

        Returns
        -------
        transformed_token : str
            Normal form of token.
        """
        assert isinstance(token, str), 'token must have str type'

        transformed_token = self._tagger.transform_token(token).get('normal_form')

        return transformed_token

    def transform_string(self, tokenized_string):
        """
        Apply lemmatizer to one tokenized string transformation.

        Parameters
        ----------
        tokenized_string : iterable of str
            List of tokens (iterable of tokens can be work too).
            Supposed to be one sentence or one document.

        Returns
        -------
        transformed_string : list of str
            List of tokens normal forms.
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
        Apply lemmatizer to collection of tokenized strings or documents.

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
