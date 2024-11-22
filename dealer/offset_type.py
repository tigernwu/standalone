


from enum import Enum


class OffsetType(Enum):
    """开平(今)"""
    Open = 0
    '''开仓'''
    Close = 1
    '''平仓'''
    CloseToday = 2
    '''平今
    上期所独有'''

    def __int__(self):
        return self.value