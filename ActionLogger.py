from Action import Action


class Action_Item:
    def __init__(self, player_id, action_type, cards):
        self.player_id = player_id
        self.action_type = action_type
        self.cards = cards


class ActionLogger:

    def __init__(self):
        self._history = []

    # def add(self, action_item):
    #    self.history.append(action_item)

    def add(self, player, action_type, cards):
        item = Action_Item(player, action_type, cards)
        self._history.append(item)

    def get_last_play(self):
        play = self._history[-1]
        return play.player_id, play.action_type, play.cards

    def get_current_rank(self):
        for item in reversed(self._history):
            if item.action_type == Action.BULLSHIT:
                return None
            if item.action_type == Action.PASS:
                pass
            if item.action_type == Action.PUT:
                return item.cards[0]

    def get_last_put_info(self):
        for item in reversed(self._history):
            if item.action_type == Action.BULLSHIT:
                return None, None
            if item.action_type == Action.PASS:
                pass
            if item.action_type == Action.PUT:
                return item.cards, item.player_id

    def get_game_history(self):
        return self._history

    def last_pass_seq(self):
        counter = 0
        for item in reversed(self._history):
            if item.action_type == Action.PASS:
                counter += 1
            else:
                return counter
