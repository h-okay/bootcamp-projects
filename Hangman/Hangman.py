import random

# choose a random word from txt file


def random_word():
    with open("randomwords.txt", "r") as file:
        all_text = file.read()
        words = list(map(str, all_text.split()))
        the_word = random.choice(words)
        return [letter for letter in the_word]


# user ready,input,again interactions
class user_interaction:
    def __init__(self):
        pass

    @staticmethod
    def user_input():
        while True:
            try:
                user_guess = input("Give me a letter: ")
                if user_guess.isalpha() and len(user_guess) == 1:
                    return user_guess.upper()
                else:
                    print("Invalid input.")
            except:
                print("Invalid input.")

    @staticmethod
    def ready():
        while True:
            try:
                again = input("Ready to play? (y/n) ")
                if again.isalpha() and again.lower() == "y":
                    return True
                elif again.isalpha() and again.lower() == "n":
                    return False
                else:
                    print("Invalid input.")
            except:
                print("Invalid input.")

    @staticmethod
    def again():
        while True:
            try:
                again = input("Do you want to play again? (y/n) ")
                if again.isalpha() and again.lower() == "y":
                    return True
                elif again.isalpha() and again.lower() == "n":
                    return False
                else:
                    print("Invalid input.")
            except:
                print("Invalid input.")


# create hidden game board
def board(arr):
    return (" __ " * len(arr)).split()


# game logic
user = user_interaction()
play = user.ready()
while play:
    # setup
    word = random_word()
    game = board(word)
    life = 10
    # play
    while ("__" in game) and (life != 0):
        print(*game)
        guess = user.user_input()

        if guess in game:
            print(f"{guess} already revealed.")
        else:
            indices = []
            for i in range(len(word)):
                if word[i] == guess:
                    indices.append(i)
            if len(indices) == 0:
                life -= 1
                print(f"Remaning life: {life}")
            for val in indices:
                game[val] = guess

    if life == 0:
        print("You are out of lives.")
        play = user.again()

    if ("__" not in game) and (life != 0):
        print("You guessed it!")
        play = user.again()

print("Thank you for playing.")
