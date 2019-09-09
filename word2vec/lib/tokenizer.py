# standard spacy utils
from spacy.lang.ru import Russian
from spacy.tokenizer import Tokenizer as SpacyBaseTokenizer
from spacy.lang.punctuation import TOKENIZER_SUFFIXES, TOKENIZER_PREFIXES, TOKENIZER_INFIXES
from spacy.lang.tokenizer_exceptions import URL_PATTERN
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex


# russian spacy regexp tokenizer
from spacy_russian_tokenizer import RussianTokenizer
from spacy_russian_tokenizer import MERGE_PATTERNS, SYNTAGRUS_RARE_CASES
from spacy_russian_tokenizer import NO_TERMINAL_PATTERNS


BASE_PREFIXES_REGEXPS = compile_prefix_regex(TOKENIZER_PREFIXES)
BASE_SUFFIXES_REGEXPS = compile_suffix_regex(TOKENIZER_SUFFIXES)
BASE_INFIXES_REGEXPS = compile_infix_regex(TOKENIZER_INFIXES)
BASE_TOKEN_MATCH = compile_infix_regex([URL_PATTERN])


class SpacyRulesRussianTokenizer():
    """
    Tokenizer based on https://github.com/aatimofeev/spacy_russian_tokenizer.git
    Tokenizer was built on spacy and use spacy standart tokenization pipeline.
    You can read more about it here:
        * https://spacy.io/usage/linguistic-features#section-tokenization
        * https://spacy.io/usage/rule-based-matching

    Installation instruction:

    1) pip install spacy
    2) pip install git+https://github.com/aatimofeev/spacy_russian_tokenizer.git
    """
    def __init__(self,
                 regexp_suffixes=BASE_SUFFIXES_REGEXPS,
                 regexp_prefixes=BASE_PREFIXES_REGEXPS,
                 regexp_infixes=BASE_INFIXES_REGEXPS,
                 regexp_base_token_matches=BASE_TOKEN_MATCH,
                 merge_patterns=tuple(MERGE_PATTERNS + SYNTAGRUS_RARE_CASES),
                 terminal_patterns=tuple(NO_TERMINAL_PATTERNS),
                 ):
        """
        Parameters
        ----------
        regexp_suffixes : list of dict
            Dict in spacy format. See above for explanation of spacy format.
        regexp_prefixes : list of dict
            Dict in spacy format.
        regexp_infixes : list of dict
            Dict in spacy format.
        regexp_base_token_matches : list of dict
            Dict in spacy format.
        merge_patterns : list of dict
            Dict in spacy format.
        terminal_patterns : list of dict
            Dict in spacy format.
        """
        merge_patterns = list(merge_patterns)
        terminal_patterns = list(terminal_patterns)

        self.nlp_pipeline = Russian()
        self.nlp_pipeline.tokenizer = self.create_custom_pretokenizer(
            nlp_model=self.nlp_pipeline,
            prefix_regexp=regexp_prefixes,
            suffix_regexp=regexp_suffixes,
            infix_regexp=regexp_infixes,
            token_match_regexp=regexp_base_token_matches,
        )

        self.tokenizer_postprocesser = RussianTokenizer(
            self.nlp_pipeline,
            merge_patterns=merge_patterns,
            terminal_patterns=terminal_patterns
        )

        self.nlp_pipeline.add_pipe(self.tokenizer_postprocesser,
                                   name='russian_tokenizer_postprocesser')

    @staticmethod
    def create_custom_pretokenizer(nlp_model, prefix_regexp, suffix_regexp,
                                   infix_regexp, token_match_regexp):
        custom_pretokenizer = SpacyBaseTokenizer(
            nlp_model.vocab,
            prefix_search=prefix_regexp.search,
            suffix_search=suffix_regexp.search,
            infix_finditer=infix_regexp.finditer,
            token_match=token_match_regexp.match,
        )
        return custom_pretokenizer

    def transform_element(self, element):
        """
        Get tokenization variant of the element.

        Parameters
        ----------
        element : str
            String, supposed to be a sentence, one document or something analogous.

        Return
        ------
        tokens_array : list of str
            Tokenized string

        """
        if not isinstance(element, str):
            raise TypeError(f"Cannot tokenize {type(element)} instead of {type('')}!")
        tokens_array = [
            token.text
            for token in self.nlp_pipeline(element)
        ]
        return tokens_array

    def transform(self, elements_collection):
        """
        Apply transformer to collection of elements (objects).

        Parameters
        ----------
        elements_collection : iterable of optional
            Collection of objects to be transformed.

        Returns
        -------
        transformed_elements : list of optional
            Collection of transformed objects.
        """
        transformed_elements = [
            self.transform_element(element)
            for element in elements_collection
        ]
        return transformed_elements
