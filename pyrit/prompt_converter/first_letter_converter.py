# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from typing import List, Optional, Union

from pyrit.prompt_converter.word_level_converter import WordLevelConverter


class FirstLetterConverter(WordLevelConverter):
    """
    Converts text by replacing each word with its first letter.
    """

    def __init__(
        self,
        *,
        indices: Optional[List[int]] = None,
        keywords: Optional[List[str]] = None,
        proportion: Optional[float] = None,
        regex: Optional[Union[str, re.Pattern]] = None,
    ):
        """
        Initializes the converter with the specified join value and selection parameters.

        This class allows for selection of words to convert based on various criteria.
        Only one selection parameter may be provided at a time (indices, keywords, proportion, or regex).
        If no selection parameter is provided, all words will be converted.

        Args:
            indices (Optional[List[int]]): Specific indices of words to convert.
            keywords (Optional[List[str]]): Keywords to select words for conversion.
            proportion (Optional[float]): Proportion of randomly selected words to convert [0.0-1.0].
            regex (Optional[Union[str, re.Pattern]]): Regex pattern to match words for conversion.
        """
        super().__init__(indices=indices, keywords=keywords, proportion=proportion, regex=regex)

    # TODO: make sure that only letters are considered
    async def convert_word_async(self, word: str) -> str:
        """Return the first character of the word (empty string if word is empty)."""
        return word[:1]
