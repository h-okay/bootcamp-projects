import pygame
import math
import random

# setup pygame
pygame.init()
WIDTH, HEIGHT = 1280, 720
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hangman")
FPS = 60
clock = pygame.time.Clock()

# fonts
LETTER_FONT = pygame.font.SysFont("comicsans", 40)
WORD_FONT = pygame.font.SysFont("comicsans", 90)
TITLE_FONT = pygame.font.SysFont("comicsans", 80)

# colors
WHITE = (180, 180, 180)
BLACK = (0, 0, 0)

# variables for buttons
RADIUS = 35
GAP = 15
A = 65


def random_word() -> str:
    """Gets a random word from the txt file."""
    with open("randomwords.txt", "r") as file:
        all_text = file.read()
        words_need_filter = list(map(str, all_text.split()))
        words = list(filter((lambda x: len(x) <= 8), words_need_filter))
        word = random.choice(words)
        return word


def message(msg):
    """Displays endgame screen"""
    pygame.time.delay(1500)
    window.fill(WHITE)
    text = WORD_FONT.render(msg, 1, BLACK)
    window.blit(
        text, (WIDTH / 2 - text.get_width() / 2,
               HEIGHT / 2 - text.get_height() / 2)
    )
    pygame.display.update()
    pygame.time.delay(3000)


def get_images() -> list:
    "Imports all related assests"
    images = []
    for i in range(7):
        image = pygame.image.load(f"../Hangman/asset/hangman{i}.png")
        images.append(image)
    return images


def get_letters() -> list:
    """Returns a list of following for pygame drawn buttons and characters: x coordinate,
    y coordinate, character on the button, boolean for visibility"""
    letters = []
    startx = round((WIDTH - (RADIUS * 2 + GAP) * 13) / 2)
    starty = 500
    for i in range(26):
        x = startx + GAP * 2 + ((RADIUS * 2 + GAP) * (i % 13))
        y = starty + (i // 13) * (GAP + RADIUS * 2)
        letters.append([x, y, chr(A + i), True])
    return letters


def get_buttons() -> list:
    """Return a list of following for custom imported buttons: x coordinate, y coordinate,
    character on the button, boolean for visibility"""
    startx = round((WIDTH - (RADIUS * 2 + GAP) * 13) / 2)
    starty = 500
    buttons = []
    for i in range(26):
        x = startx + GAP * 2 + ((RADIUS * 2 + GAP) * (i % 13))
        y = starty + (i // 13) * (GAP + RADIUS * 2)
        button = pygame.image.load(f"../Hangman/Buttons/{i}.png")
        button = pygame.transform.smoothscale(button, (160, 160))
        buttons.append([x, y, button, True])
    return buttons


def draw(word, lives, letters, buttons, images, guessed):
    """Draws screen elements"""
    window.fill(WHITE)
    text = TITLE_FONT.render("HANGMAN", 1, BLACK)
    window.blit(text, (WIDTH / 2 - text.get_width() / 2, 20))
    display_word = ""
    for letter in word:
        if letter in guessed:
            display_word += letter + " "
        else:
            display_word += "_ "
    text = WORD_FONT.render(display_word, 1, BLACK)
    window.blit(text, (500, 200))
    for letter in letters:
        x, y, ltr, visible = letter
        if visible:
            pygame.draw.circle(window, BLACK, (x, y), RADIUS, 1)
            text = LETTER_FONT.render(ltr, 1, BLACK)
            window.blit(text, (x - text.get_width() /
                        2, y - text.get_height() / 2))
    for button in buttons:
        a, b, but, visible = button
        if visible:
            window.blit(but, (a - 80, b - 80))
    window.blit(images[lives], (150, 100))
    pygame.display.update()


def main(run, lives, word, guessed, buttons, images):
    """Main game loop"""
    while run:
        clock.tick(FPS)
        draw(word, lives, letters, buttons, images, guessed)
        for event in pygame.event.get():
            # quit event
            if event.type == pygame.QUIT:
                run = False
            # click coord
            if event.type == pygame.MOUSEBUTTONDOWN:
                m_x, m_y = pygame.mouse.get_pos()
                for letter in letters:
                    x, y, ltr, visible = letter
                    if visible:
                        dis = math.sqrt((x - m_x) ** 2 + (y - m_y) ** 2)
                        if dis < RADIUS:
                            guessed.append(ltr)
                            letter[3] = False
                            buttons[letters.index(letter)][3] = False
                            if ltr not in word:
                                lives += 1

        won = True
        for letter in word:
            if letter not in guessed:
                won = False
                break
        if won:
            message("WON")
            break
        if lives == 6:
            pygame.time.delay(1500)
            answer = f"The word was {word}"
            text = WORD_FONT.render(answer, 1, BLACK)
            window.fill(WHITE)
            window.blit(
                text,
                (WIDTH / 2 - text.get_width() / 2,
                 HEIGHT / 2 - text.get_height() / 2),
            )
            pygame.display.update()
            pygame.time.delay(1500)
            message("LOST")
            break


# Playing the game
while True:
    lives = 0
    guessed = []
    word = random_word()
    letters = get_letters()
    buttons = get_buttons()
    images = get_images()
    for event in pygame.event.get():
        answer = f"Click anywhere to play"
        text = WORD_FONT.render(answer, 1, BLACK)
        window.fill(WHITE)
        window.blit(
            text, (WIDTH / 2 - text.get_width() / 2,
                   HEIGHT / 2 - text.get_height() / 2)
        )
        pygame.display.update()
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            run = True
            main(run, lives, word, guessed, buttons, images)
