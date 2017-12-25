from Action import Action

class Action_Item:
    def __init__(self, player_id, action_type, cards):

        self.player_id = player_id
        self.action_type = action_type
        self.cards = cards


class ActionLogger:

    def __init__(self):
        self.history = []

    #def add(self, action_item):
    #    self.history.append(action_item)

    def add(self, player, action_type, cards):
        item = Action_Item(player, action_type, cards)
        self.history.append(item)

    def get_last_play(self):
        play = self.history[-1]
        return play.player_id, play.action_type, play.cards

    def get_last_put_action(self):
        for item in reversed(self.history):
            if item.action_type == Action.Bullshit:
                return -1
            if item.action_type == Action.PASS:
                pass
            if item.action_type == Action.PUT:
                return item.player_id, item.action_type, item.cards