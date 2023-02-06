from SetGame import Game


tests = [["./game-images/test1.jpg", 9],
         ["./game-images/test2.jpg", 6],
         ["./game-images/test3.jpg", 9],
         ["./game-images/test4.jpg", 15]]

if __name__ == '__main__':
    for test in tests:
        filename, num_of_cards = test[0], test[1]
        game = Game(filename, num_of_cards=num_of_cards)
        game.identify_cards()
        game.find_sets()



