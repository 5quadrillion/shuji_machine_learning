# coding=utf-8


# 買いポジション
class BidPosition:
    # 買値
    __bid_price = 0
    # 損切りポイント
    __loss_cut_price = 0
    # 利確ポイント
    __profit_price = 0

    def __init__(self, bid_price, loss_cut_price, profit_price):
        self.__bid_price = bid_price
        self.__loss_cut_price = loss_cut_price
        self.__profit_price = profit_price

    # 利確か損切りポイントにかかっていたら決済する
    def settle_if_need(self, now_price):
        if now_price < self.__loss_cut_price:
            print("Loss cut!!")
            return self.__loss_cut_price - self.__bid_price
        if self.__profit_price < now_price:
            print("Set profits!!")
            return self.__profit_price - self.__bid_price
        return None


# 売りポジション
class AskPosition:
    # 売り値
    __ask_price = 0
    # 損切りポイント
    __loss_cut_price = 0
    # 利確ポイント
    __profit_price = 0

    def __init__(self, ask_price, loss_cut_price, profit_price):
        self.__ask_price = ask_price
        self.__loss_cut_price = loss_cut_price
        self.__profit_price = profit_price

    # 利確か損切りポイントにかかっていたら決済する
    def settle_if_need(self, now_price):
        if self.__loss_cut_price < now_price:
            print("Loss cut!!")
            return self.__ask_price - self.__loss_cut_price
        if now_price < self.__profit_price:
            print("Set profits!!")
            return self.__ask_price - self.__profit_price
        return None


class Positions:
    __positions = []
    __loss_cut_pips = 20
    __profit_pips = 50
    __profit_and_loss = 0

    def __init__(self, loss_cut_pips, profit_pips):
        self.__loss_cut_pips = loss_cut_pips
        self.__profit_pips = profit_pips

    def add_bid_position(self, bid_price):
        pos = BidPosition(bid_price=bid_price, loss_cut_price=(bid_price-self.__loss_cut_pips*0.01), profit_price=(bid_price+self.__profit_pips*0.01))
        self.__positions.append(pos)

    def add_ask_position(self, ask_price):
        pos = AskPosition(ask_price=ask_price, loss_cut_price=(ask_price+self.__loss_cut_pips*0.01), profit_price=(ask_price-self.__profit_pips*0.01))
        self.__positions.append(pos)

    def position_check(self, start, high, low, end):
        if start < end:
            # 本来は違うかもしれないが、start -> low -> high -> end と移り変わったと過程
            check_price = [start, low, high, end]
        else:
            # 本来は違うかもしれないが、start -> high -> low -> end と移り変わったと過程
            check_price = [start, high, low, end]
        for pos in self.__positions[:]:
            for cp in check_price:
                val = pos.settle_if_need(cp)
                if val is not None:
                    self.__profit_and_loss += val
                    self.__positions.remove(pos)

    def get_profit_and_loss(self):
        return self.__profit_and_loss
