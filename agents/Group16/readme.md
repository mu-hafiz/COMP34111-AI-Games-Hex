
These are the different cases/stages of the game:
1. Its the start of the game and we are red so we use a fair first moveset and do mcts on that.
2. Its the second turn of the game and we are blue so we need to check if we want to swap reds move by checking if it falls inside the 'winning' region.
    a. If it does we swap and we are now red.
    b. If it doesnt we dont swap and we remain blue and we need to play a good starting move. I currently dont know what the moveset should be but making a 
       connection with a wall seems good for now.
3. Its the third turn of the game and we were red but then we got swapped so we are blue and have no tiles on the board. Same idea as case 2b we need to make a good move.
4. Its our turn and the enemy has played a tile that threatens one of our  existing strong connections. We have to defend ourselves and take the other side of the connection.
    a. It is a simple formation and the enemy's move only disrupts one of our connections so we automatically select to play the other side of the connection.
    b. It is a complex formation and the enemy's move disrupts multiple connections. We run mcts on a movese of defensive plays to find the best one.
5. Its our turn and the enemy has played a tile that is not threatening any of our existing strong connections. We can do mcts on a moveset of potential connections
6. We are winning and should fill in our connections.
7. We've run out of potential connections and none of the other cases apply, fall back to all legal moves and start again