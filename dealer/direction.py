from enum import Enum


class DirectType(Enum):
    """买卖"""
    Buy = 0
    '''买'''
    Sell = 1
    '''卖'''

    def __int__(self):
        return self.value