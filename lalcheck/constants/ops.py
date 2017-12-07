PLUS = '+'
MINUS = '-'
AND = '&&'
OR = '||'
LT = '<'
LE = '<='
EQ = '=='
NEQ = '!='
GE = '>='
GT = '>'
DOT_DOT = '..'

NOT = '!'
NEG = '-'
ADDRESS = '&'
DEREF = '*'

GET_FIRST = 'GetFirst'
GET_LAST = 'GetLast'

NEW = 'new'


def get(x):
    """
    Returns the name of a record getter for the given element index.
    :param int x: The index to get.
    """
    return "Get_{}".format(x)


def updated(x):
    """
    Returns the name of a record updater for the given element index.
    :param int x: The index to update.
    """
    return "Updated_{}".format(x)
