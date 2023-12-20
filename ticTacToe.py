import numpy as np
import pickle
import random
from flask import Flask, render_template, request, jsonify
import concurrent.futures
from multiprocessing import Manager

app = Flask(__name__)

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_LAYERS = 4
MOVECOUNT=0
st= None

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_game', methods=['GET'])
def start_game():
    global st  # Verwendung der globalen Variable
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")
    st = State(p1, p2)  # Angenommen, p1 und p2 sind bereits definiert oder initialisiert
    st.play2((0,0,0))
    st.isWaiting = True
    return jsonify({'status': 'success', 'message': 'Neues Spiel gestartet'})


@app.route('/make_move', methods=['POST'])
def make_move():
    global st
    if st is None:
        return jsonify({'status': 'error', 'message': 'Spiel wurde noch nicht gestartet'})

    data = request.get_json()
    row = data['row']
    col = data['col']
    lay = data['lay']

    # Verarbeitung des menschlichen Zugs
    game_status, winner = st.play2((row, col, lay))
    if game_status in ['win', 'tie']:
        return jsonify({
            'status': 'success',
            'board': st.board.tolist(),
            'game_status': game_status,
            'winner': winner
        })

    # Verarbeitung des Computerzugs
    if not st.isWaiting:
        game_status, winner = st.play2(None)  # Annahme: Der Computer benötigt keine playerAction

    return jsonify({
        'status': 'success',
        'board': st.board.tolist(),
        'game_status': game_status,
        'winner': winner
    })



   
@app.route('/get_game_status')
def get_game_status():
    global st
    if st is None:
        return jsonify({'status': 'error', 'message': 'Spiel nicht gestartet'})
    #print("söösel", st.get_board_as_list())
    #print(jsonify({'status': 'success', 'board': st.get_board_as_list()}))
    return jsonify({'status': 'success', 'board': st.get_board_as_list()})


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS, BOARD_LAYERS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.isWaiting = False
        self.boardHash = None
        self.playerSymbol = 1
        self.states = []

    def get_board_as_list(self):
        return self.board.tolist()
    
    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS*BOARD_ROWS*BOARD_LAYERS))
        return self.boardHash
    
    def winner(self, board=None):

        if not board:
            board=self.board
        # vertical
        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                if sum(board[x, y, :]) == 4:
                    self.isEnd = True
                    return 1
                if sum(board[x, y, :]) == -4:
                    self.isEnd = True
                    return -1
        # horizontal y
        for x in range(BOARD_ROWS):
            for z in range(BOARD_LAYERS):
                if sum(board[x, :, z]) == 4:
                    self.isEnd = True
                    return 1
                if sum(board[x, :, z]) == -4:
                    self.isEnd = True
                    return -1
        # horizontal x
        for y in range(BOARD_COLS):
            for z in range(BOARD_LAYERS):
                if sum(board[:, y, z]) == 4:
                    self.isEnd = True
                    return 1
                if sum(board[:, y, z]) == -4:
                    self.isEnd = True
                    return -1

        # Diagonals from cube corner to cube corner
        diag_sum1 = sum([board[i, i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([board[i, BOARD_COLS - i - 1, i] for i in range(BOARD_COLS)])
        diag_sum3 = sum([board[i, i, BOARD_LAYERS - i - 1] for i in range(BOARD_COLS)])
        diag_sum4 = sum([board[i, BOARD_COLS - i - 1, BOARD_LAYERS - i - 1] for i in range(BOARD_COLS)])
        
        if any(val == 4 for val in [diag_sum1, diag_sum2, diag_sum3, diag_sum4]):
            self.isEnd = True
            return 1
        if any(val == -4 for val in [diag_sum1, diag_sum2, diag_sum3, diag_sum4]):
            self.isEnd = True
            return -1

        # Diagonals from cube edge to cube edge (24 of them)
        diag_sums = []
        for i in range(BOARD_COLS):
            diag_sums.append(sum([board[i, j, k] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[i, j, k] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[j, i, k] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[j, i, k] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[k, j, i] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[k, j, i] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))

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
        elif result == -1:
            self.p1.feedReward(-1)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(-0.5)
            self.p2.feedReward(0.5)
    
    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS, BOARD_LAYERS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
        global MOVECOUNT
        MOVECOUNT=0
    

        
    def play2(self, playerAction):
        global MOVECOUNT
        print("************* ACHTUNG!!! **********************")

        if not self.isWaiting:
            # Player 1 (Computer) macht einen Zug
            print("computer spielt")
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol, playerAction)
            MOVECOUNT+=1
            self.updateState(p1_action)
            self.showBoard()
            self.isWaiting = True  # Warten auf den Zug des menschlichen Spielers
        else:
            # Player 2 (Mensch) macht einen Zug
            print("mensch spielt")
            positions = self.availablePositions()
            p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol, playerAction)
            MOVECOUNT+=1
            self.updateState(p2_action)
            self.showBoard()
            self.isWaiting = False  # Computer ist wieder an der Reihe

        # Überprüfen, ob das Spiel zu Ende ist
        win = self.winner()
        if win is not None:
            self.isWaiting = True
            self.reset()
            if win == 1:
                return 'win', self.p1.name  # Gewinner ist Player 1
            elif win == -1:
                return 'win', self.p2.name  # Gewinner ist Player 2
            else:
                return 'tie', None  # Unentschieden
        return 'continue', None  # Spiel wird fortgesetzt

                

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
    

    def winner(self, board=None):

        if not board.any():
            board=self.board
        # vertical
        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                if sum(board[x, y, :]) == 4:
                    self.isEnd = True
                    return 1
                if sum(board[x, y, :]) == -4:
                    self.isEnd = True
                    return -1
        # horizontal y
        for x in range(BOARD_ROWS):
            for z in range(BOARD_LAYERS):
                if sum(board[x, :, z]) == 4:
                    self.isEnd = True
                    return 1
                if sum(board[x, :, z]) == -4:
                    self.isEnd = True
                    return -1
        # horizontal x
        for y in range(BOARD_COLS):
            for z in range(BOARD_LAYERS):
                if sum(board[:, y, z]) == 4:
                    self.isEnd = True
                    return 1
                if sum(board[:, y, z]) == -4:
                    self.isEnd = True
                    return -1

        # Diagonals from cube corner to cube corner
        diag_sum1 = sum([board[i, i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([board[i, BOARD_COLS - i - 1, i] for i in range(BOARD_COLS)])
        diag_sum3 = sum([board[i, i, BOARD_LAYERS - i - 1] for i in range(BOARD_COLS)])
        diag_sum4 = sum([board[i, BOARD_COLS - i - 1, BOARD_LAYERS - i - 1] for i in range(BOARD_COLS)])
        
        if any(val == 4 for val in [diag_sum1, diag_sum2, diag_sum3, diag_sum4]):
            self.isEnd = True
            return 1
        if any(val == -4 for val in [diag_sum1, diag_sum2, diag_sum3, diag_sum4]):
            self.isEnd = True
            return -1

        # Diagonals from cube edge to cube edge (24 of them)
        diag_sums = []
        for i in range(BOARD_COLS):
            diag_sums.append(sum([board[i, j, k] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[i, j, k] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[j, i, k] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[j, i, k] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[k, j, i] for j, k in zip(range(BOARD_COLS), range(BOARD_LAYERS))]))
            diag_sums.append(sum([board[k, j, i] for j, k in zip(range(BOARD_COLS - 1, -1, -1), range(BOARD_LAYERS))]))

        if any(val == 4 for val in diag_sums):
            self.isEnd = True
            return 1
        if any(val == -4 for val in diag_sums):
            self.isEnd = True
            return -1

        self.isEnd = False
        return None

    def countWinningMoves(self, current_board, symbol):
        count=0
        for x in range(BOARD_ROWS):
            for y in range(BOARD_COLS):
                if (sum(current_board[x, y, :])) == 3 * symbol:
                    z = next((z for z in range(BOARD_LAYERS) if current_board[x, y, z] == 0), None)
                    if z is not None:
                        action = (x, y, z)
                        count+=1

        for x in range(BOARD_ROWS):
            for z in range(BOARD_LAYERS):
                if (sum(current_board[x, :, z])) == 3 * symbol:
                    y = next((y for y in range(BOARD_COLS) if current_board[x, y, z] == 0), None)
                    if y is not None:
                        action = (x, y, z)
                        count+=1

        for y in range(BOARD_COLS):
            for z in range(BOARD_LAYERS):
                if (sum(current_board[:, y, z])) == 3 * symbol:
                    x = next((x for x in range(BOARD_ROWS) if current_board[x, y, z] == 0), None)
                    if x is not None:
                        action = (x, y, z)
                        count+=1

        # Check for winning moves in diagonals from cube corner to cube corner (4 of them)
        for i in range(BOARD_LAYERS):
            if (sum(current_board[i, i, i] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, j, j] == 0:
                        action = (j, j, j)
                        count+=1

            if (sum(current_board[i, i, BOARD_LAYERS - i - 1] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, j, BOARD_LAYERS - j - 1] == 0:
                        action = (j, j, BOARD_LAYERS - j - 1)
                        count+=1

            if (sum(current_board[i, BOARD_LAYERS - i - 1, i] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, BOARD_LAYERS - j - 1, j] == 0:
                        action = (j, BOARD_LAYERS - j - 1, j)
                        count+=1

            if (sum(current_board[i, BOARD_LAYERS - i - 1, BOARD_LAYERS - i - 1] for i in range(BOARD_LAYERS))) == 3 * symbol:
                for j in range(BOARD_LAYERS):
                    if current_board[j, BOARD_LAYERS - j - 1, BOARD_LAYERS - j - 1] == 0:
                        action = (j, BOARD_LAYERS - j - 1, BOARD_LAYERS - j - 1)
                        count+=1
                    
        for x in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for notx in range(BOARD_ROWS):
                summe1+=current_board[x,notx,notx]
            if (summe1)==3*symbol:
                for notx in range(BOARD_ROWS):
                    if current_board[x,notx,notx]==0:
                        action=(x,notx,notx)
                        count+=1
            for notx in range(BOARD_ROWS):
                summe2+=current_board[x,3-notx,notx]
            if (summe2)==3*symbol:
                for notx in range(BOARD_ROWS):
                    if current_board[x,3-notx,notx]==0:
                        action=(x,3-notx,notx)
                        count+=1
                    
        for y in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for noty in range(BOARD_ROWS):
                summe1+=current_board[noty,y,noty]
            if (summe1)==3*symbol:
                for noty in range(BOARD_ROWS):
                    if current_board[noty,y,noty]==0:
                        action=(noty,y,noty)
                        count+=1
            for noty in range(BOARD_ROWS):
                summe2+=current_board[3-noty,y,noty]
            if (summe2)==3*symbol:
                for noty in range(BOARD_ROWS):
                    if current_board[3-noty,y,noty]==0:
                        action=(3-noty,y,noty)
                        count+=1
                    
        for z in range(BOARD_ROWS):
            summe1=0
            summe2=0            
            for notz in range(BOARD_ROWS):
                summe1+=current_board[notz,notz,z]
            if (summe1)==3*symbol:
                for notz in range(BOARD_ROWS):
                    if current_board[notz,notz,z]==0:
                        action=(notz,notz,z)
                        count+=1
            for notz in range(BOARD_ROWS):
                summe2+=current_board[3-notz,notz,z]
            if (summe2)==3*symbol:
                for notz in range(BOARD_ROWS):
                    if current_board[3-notz,notz,z]==0:
                        action=(3-notz,notz,z)
                        count+=1
        return count
    
    def WinningMove(self, current_board, symbol):
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

    
    def Parent(self, current_board, positions, symbol, parentsymbol, positionsdic, depth=2, current_depth=0):
        current_depth+=1
        symbol*=-1 
        next_board = current_board.copy()
        with Manager() as manager:
            shared_dict = manager.dict(positionsdic)

            with concurrent.futures.ProcessPoolExecutor(22) as executor:
                futures = []
                for i in positions:
                    print(i)
                    positions2 = positions.copy()
                    positions2.remove(i)
                    next_board = current_board.copy()
                    next_board[i] = symbol

                    futures.append(
                        executor.submit(
                            self.MonteCarloTreeSearch,
                            next_board.copy(),
                            positions2,
                            symbol,
                            parentsymbol,
                            i,
                            shared_dict,
                            current_depth=current_depth,
                            depth=depth
                        )
                    )

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

            # Retrieve the final values from the shared dictionary
            positionsdic = dict(shared_dict)

        print(positionsdic)    
        return positionsdic

    
    def MonteCarloTreeSearch(self, current_board, positions, symbol, parentsymbol, i, positionsdic, depth, current_depth=0, forced=False):
        current_depth+=1
        symbol*=-1  
        next_board = current_board.copy()

        move=self.WinningMove(current_board.copy(), symbol)
        if move:
            print(i,"success3", symbol, current_depth)
            positionsdic[i]+=64*parentsymbol*symbol
            return

        if self.countWinningMoves(current_board, -symbol)>=2:
            #print("success1", current_depth)
            positionsdic[i]+=-64*parentsymbol*symbol

        if current_depth<=10:
            move=self.WinningMove(current_board.copy(), -symbol)        
            if move:
                #print("success2")
                #print(next_board)
                positions2=positions.copy()
                positions2.remove(move)
                next_board[move]=symbol
                #print(next_board)
                self.MonteCarloTreeSearch(next_board.copy(), positions2, symbol, parentsymbol, i, current_depth=current_depth,depth=depth, positionsdic=positionsdic, forced=True)
            
            elif forced:
                #print(current_board,"success2", current_depth)
                for j in positions:
                    positions2=positions.copy()
                    positions2.remove(j)
                    #print(next_board, current_depth)                
                    next_board[j] = symbol                
                    self.MonteCarloTreeSearch(next_board, positions2, symbol, parentsymbol, i, current_depth=current_depth,depth=depth, positionsdic=positionsdic)
                    next_board = current_board.copy()
            else:
                #print("success4")
                if current_depth <= 2:
                    for j in positions:
                        positions2=positions.copy()
                        positions2.remove(j)
                        #print(next_board, current_depth)                
                        next_board[j] = symbol
                        
                        self.MonteCarloTreeSearch(next_board, positions2, symbol, parentsymbol, i, current_depth=current_depth,depth=depth, positionsdic=positionsdic)
                        next_board = current_board.copy()      

    def chooseAction(self, positions, current_board, symbol, playerAction):
        print("count", self.countWinningMoves(current_board, -symbol))
        global MOVECOUNT
        print(MOVECOUNT)

        move=self.WinningMove(current_board, symbol)
        if move:
            return move
        
        move=self.WinningMove(current_board, -symbol)
        if move:
            return move
        
        if MOVECOUNT>=5:
            my_dict = {key: 0 for key in positions}

            positionsdic=self.Parent(current_board, positions, -symbol, symbol, positionsdic=my_dict)
            templist=[]
            for key, value in positionsdic.items():
                        if value >= max(positionsdic.values()):
                            templist.append(key)
                            
            positions=templist
                                                                                      
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

                    
                print("best_positions",best_positionlist)
                return random.choice(best_positionlist)
            
        print("max_value",value_max)
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

    def chooseAction(self, positions, current_board, symbol, playerAction):
        while True:
                print("Es wird gespielt")
                print("mensch:",playerAction)
                action = playerAction

                if action in positions:
                    return action

                else:
                    print("Invalid Input")
                    st.isWaiting = True

                action = None

    # append a hash state
    def addState(self, state):
        pass

    # at the end of the game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    app.run(debug=True)