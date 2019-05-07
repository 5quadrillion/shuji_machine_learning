# coding=utf-8
class Position:
    __pos = 0
    __max_pos = 0
    __min_pos = 0

    def __init__(self):
        pass

    def buy(self):
        self.__pos += 1
        if self.__max_pos < self.__pos:
            self.__max_pos = self.__pos

    def sell(self):
        self.__pos -= 1
        if self.__min_pos > self.__pos:
            self.__min_pos = self.__pos

    def get_max_pos(self):
        return self.__max_pos

    def get_min_pos(self):
        return self.__min_pos

    def clear(self):
        self.__pos = 0
        self.__min_pos = 0
        self.__max_pos = 0
