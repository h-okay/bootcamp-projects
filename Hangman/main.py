import math
import random
import pygame

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
        word_ = random.choice(words)
        return word_


def message(msg):
    """Displays endgame screen"""
    pygame.time.delay(1500)
    window.fill(WHITE)
    text_ = WORD_FONT.render(msg, True, BLACK)
    window.blit(
        text_, (WIDTH / 2 - text_.get_width() / 2,
                HEIGHT / 2 - text_.get_height() / 2)
    )
    pygame.display.update()
    pygame.time.delay(3000)


def get_images() -> list:
    """Import all related assets"""
    images_ = []
    for i in range(7):
        image = pygame.image.load(f"../Hangman/asset/hangman{i}.png")
        images_.append(image)
    return images_


def get_letters() -> list:
    """Returns a list of following for pygame drawn buttons and characters: x
    coordinate, y coordinate, character on the button, boolean for visibility
    """
    letters_ = []
    startx = round((WIDTH - (RADIUS * 2 + GAP) * 13) / 2)
    starty = 500
    for i in range(26):
        x = startx + GAP * 2 + ((RADIUS * 2 + GAP) * (i % 13))
        y = starty + (i // 13) * (GAP + RADIUS * 2)
        letters_.append([x, y, chr(A + i), True])
    return letters_


def get_buttons() -> list:
    """Return a list of following for custom imported buttons: x coordinate, y
    coordinate, character on the button, boolean for visibility"""
    startx = round((WIDTH - (RADIUS * 2 + GAP) * 13) / 2)
    starty = 500
    buttons_ = []
    for i in range(26):
        x = startx + GAP * 2 + ((RADIUS * 2 + GAP) * (i % 13))
        y = starty + (i // 13) * (GAP + RADIUS * 2)
        button = pygame.image.load(f"../Hangman/Buttons/{i}.png")
        button = pygame.transform.smoothscale(button, (160, 160))
        buttons_.append([x, y, button, True])
    return buttons_


def draw(draw_word, draw_lives, draw_letters, draw_buttons, draw_images,
         draw_guessed):
    """Draws screen elements"""
    window.fill(WHITE)
    text_ = TITLE_FONT.render("HANGMAN", True, BLACK)
    window.blit(text_, (WIDTH / 2 - text_.get_width() / 2, 20))
    display_word = ""
    for letter in draw_word:
        if letter in draw_guessed:
            display_word += letter + " "
        else:
            display_word += "_ "
    text_ = WORD_FONT.render(display_word, True, BLACK)
    window.blit(text_, (500, 200))
    for letter in draw_letters:
        x, y, ltr, visible = letter
        if visible:
            pygame.draw.circle(window, BLACK, (x, y), RADIUS, 1)
            text_ = LETTER_FONT.render(ltr, True, BLACK)
            window.blit(text_, (x - text_.get_width() /
                                2, y - text_.get_height() / 2))
    for button in draw_buttons:
        a, b, but, visible = button
        if visible:
            window.blit(but, (a - 80, b - 80))
    window.blit(draw_images[draw_lives], (150, 100))
    pygame.display.update()


def main(main_run, main_lives, main_word, main_guessed, main_buttons,
         main_images):
    """Main game loop"""
    while main_run:
        clock.tick(FPS)
        draw(main_word, main_lives, letters, main_buttons, main_images,
             main_guessed)
        for evnt in pygame.event.get():
            # quit event
            if evnt.type == pygame.QUIT:
                main_run = False
            # click coord
            if evnt.type == pygame.MOUSEBUTTONDOWN:
                m_x, m_y = pygame.mouse.get_pos()
                for letter in letters:
                    x, y, ltr, visible = letter
                    if visible:
                        dis = math.sqrt((x - m_x) ** 2 + (y - m_y) ** 2)
                        if dis < RADIUS:
                            main_guessed.append(ltr)
                            letter[3] = False
                            main_buttons[letters.index(letter)][3] = False
                            if ltr not in main_word:
                                main_lives += 1

        won = True
        for letter in main_word:
            if letter not in main_guessed:
                won = False
                break
        if won:
            message("WON")
            break
        if main_lives == 6:
            pygame.time.delay(1500)
            answr = f"The word was {main_word}"
            _text = WORD_FONT.render(answr, True, BLACK)
            window.fill(WHITE)
            window.blit(
                _text,
                (WIDTH / 2 - _text.get_width() / 2,
                 HEIGHT / 2 - _text.get_height() / 2),
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
        text = WORD_FONT.render(answer, True, BLACK)
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
