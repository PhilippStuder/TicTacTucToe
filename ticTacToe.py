import numpy as np
import pickle
import random
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_LAYERS = 4

@app.route('/')
def index():
    return render_template('index.html')

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS, BOARD_LAYERS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.states = []
    
    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS*BOARD_ROWS*BOARD_LAYERS))
        return self.boardHash
    
    def winner(self):
        # vertical
        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                if sum(self.board[x, y, :]) == 4:
                    self.isEnd = True
                    return 1
                if sum(self.board[x, y, :]) == -4:
                    self.isEnd = True
                    return -1
        # horizontal y
        for x in range(BOARD_ROWS):
            for z in range(BOARD_LAYERS):
                if sum(self.board[x, :, z]) == 4:
                    self.isEnd = True
                    return 1
                if sum(self.board[x, :, z]) == -4:
                    self.isEnd = True
                    return -1
        # horizontal x
        for y in range(BOARD_COLS):
            for z in range(BOARD_LAYERS):
                if sum(self.board[:, y, z]) == 4:
                    self.isEnd = True
                    return 1
                if sum(self.board[:, y, z]) == -4:
                    self.isEnd = True
                    return -1

        # Diagonals from cube corner to cube corner
        diag_sum1 = sum([self.board[i, i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1, i] for i in range(BOARD_COLS)])
        diag_sum3 = sum([self.board[i, i, BOARD_LAYERS - i - 1] for i in range(BOARD_COLS)])
        diag_sum4 = sum([self.board[i, BOARD_COLS - i - 1, BOARD_LAYERS - i - 1] for i in range(BOARD_COLS)])
        
        if any(val == 4 for val in [diag_sum1, diag_sum2, diag_sum3, diag_sum4]):
            self.isEnd = True
            return 1
        if any(val == -4 for val in [diag_sum1, diag_sum2, diag_sum3, diag_sum4]):
            self.isEnd = True
            return -1

        # Diagonals from cube edge to cube edge (24 of them)
        diag_sums = []
        for i in range(BOARD_COLS):
            diag_sums.append(sum([self.board[i, j, k] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([self.board[i, j, k] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))
            diag_sums.append(sum([self.board[j, i, k] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([self.board[j, i, k] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))
            diag_sums.append(sum([self.board[k, j, i] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([self.board[k, j, i] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))

        if any(val == 4 for val in diag_sums):
            self.isEnd = True
            return 1
        if any(val == -4 for val in diag_sums):
            self.isEnd = True
            return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            print("tie")
            print(len(self.availablePositions()))
            self.isEnd = True
            return 0

        # not end
        self.isEnd = False
        return None

    
    def availablePositions(self):
        positions = []
        
        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                for z in range(BOARD_LAYERS):
                    if self.board[x,y,z] == 0:
                        positions.append((x,y,z))  # need to be tuple
        return positions
    
    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
    
    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(-1)
            #print("reward got fed to 1")
        elif result == -1:
            self.p1.feedReward(-1)
            self.p2.feedReward(1)
            #print("reward got fed to -1")
        else:
            self.p1.feedReward(-0.5)
            self.p2.feedReward(0.5)
            #print("reward got fed to both")
    
    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS, BOARD_LAYERS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
    
    def play(self, rounds=100):
        

        for i in range(rounds):

            if i%100 == 0:
                print("Rounds {}".format(i))
                p1.savePolicy()
                p2.savePolicy()
                print("saved")
                p1.loadPolicy("policy_p1")
                p2.loadPolicy("policy_p2")

            while not self.isEnd:
                
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end
                self.states.append(board_hash)
                
                win = self.winner()
                if win is not None:
                    print("player1 wins", len(positions))
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)
                    
                    win = self.winner()
                    if win is not None:
                        print("player2 wins", len(positions))
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
    
    # play with human
    def play2(self):
        self.showBoard()
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        print("#########################")   
        print(self.board) 


class Player:
    def __init__(self, name, exp_rate=0.1):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.5
        self.states_value = {}  # state -> value
    
    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS*BOARD_ROWS*BOARD_LAYERS))
        return boardHash
    
    def chooseAction(self, positions, current_board, symbol):
        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                if (sum(current_board[x, y, :])) == 3 * symbol:
                    z = next((z for z in range(BOARD_LAYERS) if current_board[x, y, z] == 0), None)
                    if z is not None:
                        action = (x, y, z)
                        return action

        for x in range(BOARD_ROWS):
            for z in range(BOARD_LAYERS):
                if (sum(current_board[x, :, z])) == 3 * symbol:
                    y = next((y for y in range(BOARD_COLS) if current_board[x, y, z] == 0), None)
                    if y is not None:
                        action = (x, y, z)
                        return action

        for y in range(BOARD_COLS):
            for z in range(BOARD_LAYERS):
                if (sum(current_board[:, y, z])) == 3 * symbol:
                    x = next((x for x in range(BOARD_ROWS) if current_board[x, y, z] == 0), None)
                    if x is not None:
                        action = (x, y, z)
                        return action

        # Check for winning moves in diagonals from cube corner to cube corner (4 of them)
        for i in range(BOARD_LAYERS):
            if (sum(current_board[i, i, i] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, j, j] == 0:
                        action = (j, j, j)
                        return action

            if (sum(current_board[i, i, BOARD_LAYERS - i - 1] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, j, BOARD_LAYERS - j - 1] == 0:
                        action = (j, j, BOARD_LAYERS - j - 1)
                        return action

            if (sum(current_board[i, BOARD_LAYERS - i - 1, i] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, BOARD_LAYERS - j - 1, j] == 0:
                        action = (j, BOARD_LAYERS - j - 1, j)
                        return action

            if (sum(current_board[i, BOARD_LAYERS - i - 1, BOARD_LAYERS - i - 1] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, BOARD_LAYERS - j - 1, BOARD_LAYERS - j - 1] == 0:
                        action = (j, BOARD_LAYERS - j - 1, BOARD_LAYERS - j - 1)
                        return action
                    
        for x in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for notx in range(BOARD_ROWS):
                summe1+=current_board[x,notx,notx]
            if (summe1)==3*symbol:
                for notx in range(BOARD_ROWS):
                    if current_board[x,notx,notx]==0:
                        action=(x,notx,notx)
                        return action
            for notx in range(BOARD_ROWS):
                summe2+=current_board[x,3-notx,notx]
            if (summe2)==3*symbol:
                for notx in range(BOARD_ROWS):
                    if current_board[x,3-notx,notx]==0:
                        action=(x,3-notx,notx)
                        return action
                    
        for y in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for noty in range(BOARD_ROWS):
                summe1+=current_board[noty,y,noty]
            if (summe1)==3*symbol:
                for noty in range(BOARD_ROWS):
                    if current_board[noty,y,noty]==0:
                        action=(noty,y,noty)
                        return action
            for noty in range(BOARD_ROWS):
                summe2+=current_board[3-noty,y,noty]
            if (summe2)==3*symbol:
                for noty in range(BOARD_ROWS):
                    if current_board[3-noty,y,noty]==0:
                        action=(3-noty,y,noty)
                        return action
                    
        for z in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for notz in range(BOARD_ROWS):
                summe1+=current_board[notz,notz,z]
            if (summe1)==3*symbol:
                for notz in range(BOARD_ROWS):
                    if current_board[notz,notz,z]==0:
                        action=(notz,notz,z)
                        return action
            for notz in range(BOARD_ROWS):
                summe2+=current_board[3-notz,notz,z]
            if (summe2)==3*symbol:
                for notz in range(BOARD_ROWS):
                    if current_board[3-notz,notz,z]==0:
                        action=(3-notz,notz,z)
                        return action

             #########################################################################       

        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                if (sum(current_board[x, y, :])) == 3 * (-symbol):
                    z = next((z for z in range(BOARD_LAYERS) if current_board[x, y, z] == 0), None)
                    if z is not None:
                        action = (x, y, z)
                        return action

        for x in range(BOARD_ROWS):
            for z in range(BOARD_LAYERS):
                if (sum(current_board[x, :, z])) == 3 * (-symbol):
                    y = next((y for y in range(BOARD_COLS) if current_board[x, y, z] == 0), None)
                    if y is not None:
                        action = (x, y, z)
                        return action

        for y in range(BOARD_COLS):
            for z in range(BOARD_LAYERS):
                if (sum(current_board[:, y, z])) == 3 * (-symbol):
                    x = next((x for x in range(BOARD_ROWS) if current_board[x, y, z] == 0), None)
                    if x is not None:
                        action = (x, y, z)
                        return action

        # Check for winning moves in diagonals from cube corner to cube corner (4 of them)
        for i in range(BOARD_LAYERS):
            if (sum(current_board[i, i, i] for i in range(BOARD_LAYERS))) == 3 * (-symbol):
                for j in range(BOARD_LAYERS):
                    if current_board[j, j, j] == 0:
                        action = (j, j, j)
                        return action

            if (sum(current_board[i, i, BOARD_LAYERS - i - 1] for i in range(BOARD_LAYERS))) == 3 * (-symbol):
                for j in range(BOARD_LAYERS):
                    if current_board[j, j, BOARD_LAYERS - j - 1] == 0:
                        action = (j, j, BOARD_LAYERS - j - 1)
                        return action

            if (sum(current_board[i, BOARD_LAYERS - i - 1, i] for i in range(BOARD_LAYERS))) == 3 * (-symbol):
                for j in range(BOARD_LAYERS):
                    if current_board[j, BOARD_LAYERS - j - 1, j] == 0:
                        action = (j, BOARD_LAYERS - j - 1, j)
                        return action

            if (sum(current_board[i, BOARD_LAYERS - i - 1, BOARD_LAYERS - i - 1] for i in range(BOARD_LAYERS))) == 3 * (-symbol):
                for j in range(BOARD_LAYERS):
                    if current_board[j, BOARD_LAYERS - j - 1, BOARD_LAYERS - j - 1] == 0:
                        action = (j, BOARD_LAYERS - j - 1, BOARD_LAYERS - j - 1)
                        return action
                    
        for x in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for notx in range(BOARD_ROWS):
                summe1+=current_board[x,notx,notx]
            if (summe1)==3*(-symbol):
                for notx in range(BOARD_ROWS):
                    if current_board[x,notx,notx]==0:
                        action=(x,notx,notx)
                        return action
            for notx in range(BOARD_ROWS):
                summe2+=current_board[x,3-notx,notx]
            if (summe2)==3*(-symbol):
                for notx in range(BOARD_ROWS):
                    if current_board[x,3-notx,notx]==0:
                        action=(x,3-notx,notx)
                        return action
                    
        for y in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for noty in range(BOARD_ROWS):
                summe1+=current_board[noty,y,noty]
            if (summe1)==3*(-symbol):
                for noty in range(BOARD_ROWS):
                    if current_board[noty,y,noty]==0:
                        action=(noty,y,noty)
                        return action
            for noty in range(BOARD_ROWS):
                summe2+=current_board[3-noty,y,noty]
            if (summe2)==3*(-symbol):
                for noty in range(BOARD_ROWS):
                    if current_board[3-noty,y,noty]==0:
                        action=(3-noty,y,noty)
                        return action
                    
        for z in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for notz in range(BOARD_ROWS):
                summe1+=current_board[notz,notz,z]
            if (summe1)==3*(-symbol):
                for notz in range(BOARD_ROWS):
                    if current_board[notz,notz,z]==0:
                        action=(notz,notz,z)
                        return action
            for notz in range(BOARD_ROWS):
                summe2+=current_board[3-notz,notz,z]
            if (summe2)==3*(-symbol):
                for notz in range(BOARD_ROWS):
                    if current_board[3-notz,notz,z]==0:
                        action=(3-notz,notz,z)
                        return action

        ####################################################################
                                                                                      
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]

        
        
        else:
            valuelist=[]
            value_max = -999
            positionsdic={}

            for p in positions:
                x, y, z = p
                next_board = current_board.copy()
                next_board[(x, y, z)] = symbol  # Update the position with the player symbol

                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                
                if value >= value_max:
                    valuelist.append((x,y,z))
                    value_max = value
                    action = (x, y, z)
                if value <0:
                    positions.remove(p)
            #print(len(positions), value_max)

            if value_max==0:
                #index=np.random.choice(len(valuelist))
                #action=valuelist[index]

                for p in positions:
                    x, y, z = p
                    templist1=[]
                    templist2=[]
                    templist3=[]
                    templist4=[]
                    templist5=[]
                    templist6=[]
                    templist7=[]
                    templist8=[]
                    templist9=[]
                    templist10=[]
                    templist11=[]
                    templist12=[]
                    templist13=[]
                    best_positionlist=[]

                    positionvalue=0

                    for i in range(-3,4):
                        if x-i>=0 and x-i<4 and y-i>=0 and y-i<4 and z-i >=0 and z-i<4 and current_board[x-i,y-i,z-i] != -symbol:
                            templist1.append((x-i,y-i,z-i))
                    if len(templist1)==4:
                        positionvalue+=1
                        for j in templist1:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if x-i>=0 and x-i<4 and y-i>=0 and y-i<4 and z+i >=0 and z+i<4 and current_board[x-i,y-i,z+i] != -symbol:
                            templist2.append((x-i,y-i,z+i))
                    if len(templist2)==4:
                        positionvalue+=1
                        for j in templist2:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if x-i>=0 and x-i<4 and y+i>=0 and y+i<4 and z-i >=0 and z-i<4 and current_board[x-i,y+i,z-i] != -symbol:
                            templist3.append((x-i,y+i,z-i))
                    if len(templist3)==4:
                        positionvalue+=1
                        for j in templist3:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if x-i>=0 and x-i<4 and y+i>=0 and y+i<4 and z+i >=0 and z+i<4 and current_board[x-i,y+i,z+i] != -symbol:
                            templist4.append((x-i,y+i,z+i))
                    if len(templist4)==4:
                        positionvalue+=1
                        for j in templist4:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if z-i >=0 and z-i<4 and current_board[x,y,z-i] != -symbol:
                            templist5.append((x,y,z-i))
                    if len(templist5)==4:
                        positionvalue+=1
                        for j in templist5:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if x-i >=0 and x-i<4 and current_board[x-i,y,z] != -symbol:
                            templist6.append((x-i,y,z))
                    if len(templist6)==4:
                        positionvalue+=1
                        for j in templist6:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if y-i >=0 and y-i<4 and current_board[x,y-i,z] != -symbol:
                            templist7.append((x,y-i,z))
                    if len(templist7)==4:
                        positionvalue+=1
                        for j in templist7:
                            if current_board[j]==symbol:
                                positionvalue+=0.26
                    
                    for i in range(-3,4):
                        if x-i>=0 and x-i<4 and z-i >=0 and z-i<4 and current_board[x-i,y,z-i] != -symbol:
                            templist8.append((x-i,y,z-i))
                    if len(templist8)==4:
                        positionvalue+=1
                        for j in templist8:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if x-i>=0 and x-i<4 and z+i >=0 and z+i<4 and current_board[x-i,y,z+i] != -symbol:
                            templist9.append((x-i,y,z+i))
                    if len(templist9)==4:
                        positionvalue+=1
                        for j in templist9:
                            if current_board[j]==symbol:
                                positionvalue+=0.26
                    
                    for i in range(-3,4):
                        if y-i>=0 and y-i<4 and z-i >=0 and z-i<4 and current_board[x,y-i,z-i] != -symbol:
                            templist10.append((x,y-i,z-i))
                    if len(templist10)==4:
                        positionvalue+=1
                        for j in templist10:
                            if current_board[j]==symbol:
                                positionvalue+=0.26
                    
                    for i in range(-3,4):
                        if y-i>=0 and y-i<4 and z+i >=0 and z+i<4 and current_board[x,y-i,z+i] != -symbol:
                            templist11.append((x,y-i,z+i))
                    if len(templist11)==4:
                        positionvalue+=1
                        for j in templist11:
                            if current_board[j]==symbol:
                                positionvalue+=0.26
                    
                    for i in range(-3,4):
                        if y-i>=0 and y-i<4 and x-i >=0 and x-i<4 and current_board[x-i,y-i,z] != -symbol:
                            templist12.append((x-i,y-i,z))
                    if len(templist12)==4:
                        positionvalue+=1
                        for j in templist12:
                            if current_board[j]==symbol:
                                positionvalue+=0.26

                    for i in range(-3,4):
                        if y+i>=0 and y+i<4 and x-i >=0 and x-i<4 and current_board[x-i,y+i,z] != -symbol:
                            templist13.append((x-i,y+i,z))
                    if len(templist13)==4:
                        positionvalue+=1
                        for j in templist13:
                            if current_board[j]==symbol:
                                positionvalue+=0.26            
                    
                    positionsdic.update({p:positionvalue})

                    for key, value in positionsdic.items():
                        if value >= max(positionsdic.values()):
                            best_positionlist.append(key)

                    
                #print(best_positionlist)
                return random.choice(best_positionlist)
            
        #print(value_max)
        return action
    
    # append a hash state
    def addState(self, state):
        self.states.append(state)
    
    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr*(self.decay_gamma*reward - self.states_value[st])
            #reward = self.states_value[st]
            reward=reward*self.decay_gamma
            
    def reset(self):
        self.states = []
        
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file,'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions, current_board, symbol):
        while True:
                lay = int(input("Input your action lay:"))
                row = int(input("Input your action row:"))
                col = int(input("Input your action col:"))
                action = (lay, row, col)

                if action in positions:
                    return action
                else:
                    print("Invalid Input")

    # append a hash state
    def addState(self, state):
        pass

    # at the end of the game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    # training
    # p1 = Player("p1")
    # #p1.loadPolicy("policy_p1")
    # p2 = Player("p2")
    # #p2.loadPolicy("policy_p2")

    # st = State(p1, p2)
    # print("training...")
    # st.play(10000)

    # p1.savePolicy()
    # p2.savePolicy()
    # print("saved successfully")

    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")

    st = State(p2, p1)
    st.play2()