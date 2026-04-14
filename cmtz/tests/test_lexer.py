"""Test the lexer TOKEN_SPEC ordering fix.

v1 bug: MODE ('real'|'imag'|'mag'|'trace') was listed after ID
in TOKEN_SPEC. Since the regex alternation matches left-to-right,
'real' tokenized as ID('real') not MODE('real') when used as a
statement keyword. This caused parse failures.
"""

import pytest
from cmtz.lexer import tokenize, LexError


class TestLexerModeOrdering:
    def test_real_is_mode(self):
        tokens = tokenize("measure(x, real) as out;")
        types = [t.type for t in tokens if t.type != 'EOF']
        assert 'MODE' in types
        mode_tok = [t for t in tokens if t.type == 'MODE'][0]
        assert mode_tok.value == 'real'

    def test_imag_is_mode(self):
        tokens = tokenize("measure(x, imag) as out;")
        mode_tok = [t for t in tokens if t.type == 'MODE'][0]
        assert mode_tok.value == 'imag'

    def test_mod_p_is_mode(self):
        tokens = tokenize("measure(x, mod_p) as out;")
        mode_tok = [t for t in tokens if t.type == 'MODE'][0]
        assert mode_tok.value == 'mod_p'

    def test_keywords_before_id(self):
        tokens = tokenize("field(7); embed(0, 0); roots(7) as r;")
        types = [t.type for t in tokens if t.type != 'EOF']
        assert 'KW_FIELD' in types
        assert 'KW_EMBED' in types
        assert 'KW_ROOTS' in types

    def test_id_still_works(self):
        tokens = tokenize("measure(myvar, real) as out;")
        id_tok = [t for t in tokens if t.type == 'ID'][0]
        assert id_tok.value == 'myvar'


class TestLexerLineCol:
    def test_single_line(self):
        tokens = tokenize("embed(0, 0);")
        assert tokens[0].line == 1
        assert tokens[0].col == 1

    def test_multiline(self):
        src = "embed(0, 0);\nrotate(a, b, 3);"
        tokens = tokenize(src)
        rotate_tok = [t for t in tokens if t.type == 'KW_ROTATE'][0]
        assert rotate_tok.line == 2

    def test_comment_skipped(self):
        tokens = tokenize("// comment\nembed(0, 0);")
        non_eof = [t for t in tokens if t.type != 'EOF']
        assert non_eof[0].type == 'KW_EMBED'
        assert non_eof[0].line == 2

    def test_dash_comment(self):
        tokens = tokenize("-- comment\nembed(0, 0);")
        non_eof = [t for t in tokens if t.type != 'EOF']
        assert non_eof[0].type == 'KW_EMBED'

    def test_bad_char_raises(self):
        with pytest.raises(LexError):
            tokenize("embed(0, 0) @")
