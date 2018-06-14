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

CALL = "Call"
UPDATED = "Updated"

IMAGE = "Image"

COPY_OFFSET = "CopyOffset"

GET_MODEL = 'GetModel'


class _IndexedName(object):
    def __init__(self, index, frmt):
        self.index = index
        self.frmt = frmt

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.index == other.index

    def __hash__(self):
        return hash(self.index)

    def __str__(self):
        return self.frmt.format(self.index)


class GetName(_IndexedName):
    def __init__(self, index):
        super(GetName, self).__init__(index, "Get_{}")


class UpdatedName(_IndexedName):
    def __init__(self, index):
        super(UpdatedName, self).__init__(index, "Updated_{}")


class OffsetName(_IndexedName):
    def __init__(self, index):
        super(OffsetName, self).__init__(index, "Offset_{}")
