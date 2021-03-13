#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tkinter import Tk, messagebox, Canvas, PhotoImage, Button, DISABLED, NORMAL, HIDDEN
from os import path
from itertools import product
from typing import Tuple, Optional, Iterable
import numpy as np
from go_types import *
from go_board import *

resource_dir = "./img"


class BoardCanvas(Canvas, GoBoard):
    KernelMap = {9: 2, 13: 3, 19: 6}
    OffsetMap = {9: 1, 13: 2 / 3, 19: 4 / 9}
    PlayerName = ["black", "white"]

    def __init__(self, shape=19, master=None, scale=1.8, meter=400, pad=20):
        Canvas.__init__(self, master, bg='#369', bd=0, width=1.5 * meter * scale, height=meter * scale)
        GoBoard.__init__(self, shape=shape)
        self.scale, self.shape, self.meter, self.pad = scale, shape, meter, pad
        self.place(x=0, y=0)
        self.create_rectangle(0, 0, meter * scale, meter * scale, fill='#c51')
        # 刻画棋盘线及九个点
        # 先画外框粗线
        self.create_rectangle(pad * scale, pad * scale, (meter - pad) * scale, (meter - pad) * scale, width=3)
        unit = (meter - 2 * pad) * scale / (shape - 1)
        self.unit = unit
        # 棋盘上的九个定位点，以中点为模型，移动位置，以作出其余八个点
        radius = 2 * scale
        core = meter / 2 * scale
        for m, n in product((-1, 0, 1), (-1, 0, 1)):
            original = self.create_oval(core - radius,
                                        core - radius,
                                        core + radius,
                                        core + radius, fill='#000')
            self.move(original,
                      m * unit * BoardCanvas.KernelMap[shape],
                      n * unit * BoardCanvas.KernelMap[shape])

        for i in range(1, shape - 1):
            self.create_line(pad * scale, pad * scale + i * unit, (meter - pad) * scale,
                             pad * scale + i * unit, width=2)
            self.create_line(pad * scale + i * unit, pad * scale, pad * scale + i * unit,
                             (meter - pad) * scale, width=2)

        photoBD = PhotoImage(file=path.join(resource_dir, "BD-%d.png" % shape))
        photoWD = PhotoImage(file=path.join(resource_dir, "WD-%d.png" % shape))
        photoBU = PhotoImage(file=path.join(resource_dir, "BU-%d.png" % shape))
        photoWU = PhotoImage(file=path.join(resource_dir, "WU-%d.png" % shape))
        self.photoWBU = (photoBU, photoWU)
        self.photoWBD = (photoBD, photoWD)
        self.photoW = PhotoImage(file=path.join(resource_dir, "W.png"))
        self.photoB = PhotoImage(file=path.join(resource_dir, "B.png"))
        self.pW = self.create_image(1.25 * meter * scale + 11, 0.16 * meter * scale, image=self.photoW)
        self.pB = self.create_image(1.25 * meter * scale - 11, 0.16 * meter * scale, image=self.photoB)
        self.addtag_withtag('image', self.pW)
        self.addtag_withtag('image', self.pB)
        self.player_stone_image = self.create_image(0, 0, state=HIDDEN, image=photoBU), \
                                  self.create_image(0, 0, state=HIDDEN, image=photoWU)
        self.__stones = np.empty((shape, shape), dtype=np.object)
        self.__image_signs = {}
        self.__stop_hover = False
        self.stop = False
        self.start_hover()
        self.__hover_pos = None
        self.bind('<Button-1>', self.on_click)

    @property
    def __player(self) -> int:
        return self._next_player.value

    def hover(self, event):
        if not self.stop and not self.__stop_hover:
            _min = self.pad * self.scale
            _max = (self.meter - self.pad) * self.scale
            if _min < event.x < _max and _min < event.y < _max:
                hover_pos = self.screen_to_grid((event.x, event.y))
                if self.__stones.item(hover_pos) is None:
                    self.__hover_pos = hover_pos
                    x, y = self.grid_to_screen(hover_pos)
                    hover_image = self.player_stone_image[self.__player]
                    other_image = self.player_stone_image[1 - self.__player]
                    self.coords(hover_image, x + 22 * BoardCanvas.OffsetMap[self.shape],
                                y - 27 * BoardCanvas.OffsetMap[self.shape])
                    self.itemconfig(other_image, state=HIDDEN)
                    self.itemconfig(hover_image, state=NORMAL)
        else:
            self.itemconfig(self.player_stone_image[0], state=HIDDEN)
            self.itemconfig(self.player_stone_image[1], state=NORMAL)

    def stop_hover(self):
        self.__stop_hover = True
        self.unbind("<Motion>")
        self.itemconfig(self.player_stone_image[0], state=HIDDEN)
        self.itemconfig(self.player_stone_image[1], state=HIDDEN)

    def start_hover(self):
        self.__stop_hover = False
        self.bind("<Motion>", self.hover)

    def pause_hover(self, stop=True):
        self.__stop_hover = stop
        if stop:
            self.itemconfig(self.player_stone_image[0], state=HIDDEN)
            self.itemconfig(self.player_stone_image[1], state=HIDDEN)

    def screen_to_grid(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        _min = self.pad * self.scale
        return round((pos[0] - _min) / self.unit), round((pos[1] - _min) / self.unit)

    def grid_to_screen(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        _min = self.pad * self.scale
        return int(_min + pos[0] * self.unit), int(_min + pos[1] * self.unit)

    def __place_stone(self, pos: GoPoint, player: GoPlayer = GoPlayer.none):
        stone = self.__stones.item(pos)
        if stone is not None:
            self.delete(stone)
        x, y = self.grid_to_screen(pos)
        if player == GoPlayer.none:
            stone = None
        else:
            stone = self.create_image(x + 4 * BoardCanvas.OffsetMap[self.shape],
                                      y - 5 * BoardCanvas.OffsetMap[self.shape],
                                      image=self.photoWBD[player.value])
            self.addtag_withtag('stone', stone)
        self.__stones.itemset(pos, stone)

    def _place_stone(self, pos: GoPoint, player: GoPlayer = GoPlayer.none):
        GoBoard._place_stone(self, pos, player)
        self.__place_stone(pos, player)

    def is_on_grid(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.shape and 0 <= pos[1] < self.shape

    def on_click(self, event):
        if not self.stop and self.__hover_pos:
            self.clear_signs()
            try:
                self.play(self.__hover_pos)
                self.add_sign(self.__hover_pos)
            except GoIllegalActionError as e:
                self.warning("illegal action", e)
                print(e.details)

    def add_sign(self, pos: Tuple[int, int], color="#3ae"):
        self.remove_sign(pos)
        x, y = self.grid_to_screen(pos)
        sign = self.create_oval(
            x + 0.5 * self.unit,
            y + 0.5 * self.unit,
            x - 0.5 * self.unit,
            y - 0.5 * self.unit, width=3,
            outline=color)
        self.addtag_withtag("sign", sign)
        self.__image_signs[pos] = sign

    def remove_sign(self, pos: Tuple[int, int]):
        if pos in self.__image_signs:
            self.delete(self.__image_signs[pos])
            del self.__image_signs[pos]

    def clear_signs(self):
        for pos in self.__image_signs:
            self.delete(self.__image_signs[pos])
        self.__image_signs = {}

    def warning(self, title: str, msg: str):
        tmp = self.__stop_hover
        self.pause_hover(True)
        messagebox.showwarning(title, msg, parent=self)
        self.pause_hover(tmp)

        '''
        def getDown(self, event):
            if not self.stop:
                # 先找到最近格点
                if (20 * self.scale - self.dd * 0.4 < event.x < self.dd * 0.4 + 380 * self.size) and (
                        20 * self.size - self.dd * 0.4 < event.y < self.dd * 0.4 + 380 * self.size):
                    dx = (event.x - 20 * self.size) % self.dd
                    dy = (event.y - 20 * self.size) % self.dd
                    x = int((event.x - 20 * self.size - dx) / self.dd + round(dx / self.dd) + 1)
                    y = int((event.y - 20 * self.size - dy) / self.dd + round(dy / self.dd) + 1)
                    # 判断位置是否已经被占据
                    if self.positions[y][x] == 0:
                        # 未被占据，则尝试占据，获得占据后能杀死的棋子列表
                        self.positions[y][x] = self.present + 1
                        self.image_added = self.canvas_bottom.create_image(
                            event.x - dx + round(dx / self.dd) * self.dd + 4 * self.p,
                            event.y - dy + round(dy / self.dd) * self.dd - 5 * self.p,
                            image=self.photoWBD_list[self.present])
                        self.canvas_bottom.addtag_withtag('image', self.image_added)
                        # 棋子与位置标签绑定，方便“杀死”
                        self.canvas_bottom.addtag_withtag('position' + str(x) + str(y), self.image_added)
                        deadlist = self.get_deadlist(x, y)
                        self.kill(deadlist)
                        # 判断是否重复棋局
                        if not self.last_2_positions == self.positions:
                            # 判断是否属于有气和杀死对方其中之一
                            if len(deadlist) > 0 or self.if_dead([[x, y]], self.present + 1, [x, y]) == False:
                                # 当不重复棋局，且属于有气和杀死对方其中之一时，落下棋子有效
                                if not self.regretchance == 1:
                                    self.regretchance += 1
                                else:
                                    self.regretButton['state'] = NORMAL
                                self.last_3_positions = copy.deepcopy(self.last_2_positions)
                                self.last_2_positions = copy.deepcopy(self.last_1_positions)
                                self.last_1_positions = copy.deepcopy(self.positions)
                                # 删除上次的标记，重新创建标记
                                self.delete('image_added_sign')
                                self.image_added_sign = self.canvas_bottom.create_oval(
                                    event.x - dx + round(dx / self.dd) * self.dd + 0.5 * self.dd,
                                    event.y - dy + round(dy / self.dd) * self.dd + 0.5 * self.dd,
                                    event.x - dx + round(dx / self.dd) * self.dd - 0.5 * self.dd,
                                    event.y - dy + round(dy / self.dd) * self.dd - 0.5 * self.dd, width=3,
                                    outline='#3ae')
                                self.addtag_withtag('image', self.image_added_sign)
                                self.addtag_withtag('image_added_sign', self.image_added_sign)
                                if self.present == 0:
                                    self.create_pW()
                                    self.del_pB()
                                    self.present = 1
                                else:
                                    self.create_pB()
                                    self.del_pW()
                                    self.present = 0
                            else:
                                # 不属于杀死对方或有气，则判断为无气，警告并弹出警告框
                                self.positions[y][x] = 0
                                self.canvas_bottom.delete('position' + str(x) + str(y))
                                self.bell()
                                self.showwarningbox('无气', "你被包围了！")
                        else:
                            # 重复棋局，警告打劫
                            self.positions[y][x] = 0
                            self.canvas_bottom.delete('position' + str(x) + str(y))
                            self.recover(deadlist, (1 if self.present == 0 else 0))
                            self.bell()
                            self.showwarningbox("打劫", "此路不通！")
                    else:
                        # 覆盖，声音警告
                        self.bell()
                else:
                    # 超出边界，声音警告
                    self.bell()
        '''


class Example(Tk):
    def __init__(self, shape=19, scale=1.8, meter=400, pad=20):
        super().__init__("Chess")
        self.geometry(str(int(1.5 * meter * scale)) + 'x' + str(int(meter * scale)))
        self.board = BoardCanvas(shape=shape, master=self, scale=scale, meter=meter, pad=pad)
        # 几个功能按钮
        button_x = 1.2 * meter * scale
        button_y = [(meter / 2 + i * meter / 16) * scale for i in range(6)]
        button_width = int(meter / 64 * scale)
        self.startButton = Button(self, width=button_width, text='Start', command=self.start)
        self.startButton.place(x=button_x, y=button_y[0])

        self.passButton = Button(self, width=button_width, text='Pass', command=self._pass)
        self.passButton.place(x=button_x, y=button_y[1])

        self.regretButton = Button(self, width=button_width, text='Regret', command=self.regret)
        self.regretButton.place(x=button_x, y=button_y[2])
        self.regretButton['state'] = DISABLED

        self.replayButton = Button(self, width=button_width, text='Replay', command=self.reload)
        self.replayButton.place(x=button_x, y=button_y[3])

        self.quitButton = Button(self, width=button_width, text='Quit', command=self.destroy)
        self.quitButton.place(x=button_x, y=button_y[4])

    def reload(self):
        pass

    def _pass(self):
        pass

    def regret(self):
        pass

    def start(self):
        pass


if __name__ == '__main__':
    # 循环，直到不切换游戏模式
    app = Example()
    # app.title('围棋')
    app.mainloop()
    # print(np.zeros((2, 2), dtype=np.object))
