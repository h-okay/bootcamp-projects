import pygame

# init
pygame.init()
WIDTH, HEIGHT = 1280, 720
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hangman")

# load img
images = []
for i in range(7):
    image = pygame.image.load(
        f"../Hangman/asset/hangman{i}.png")
    images.append(image)

# button var
RADIUS = 30
GAP = 20
letters = []
startx = round((WIDTH-(RADIUS * 2 + GAP) * 15)/2)
starty = 450
for i in range(26):
    x = startx + GAP * 2 + ((RADIUS * 2 + GAP) * (i % 13))
    y = starty + (i // 13) * (GAP + RADIUS * 2)
    letters.append([x, y])

buttons = []
for i in range(26):
    button = pygame.image.load(f"../Hangman/Buttons/{i}.png")
    button = pygame.transform.smoothscale(button, (150, 150))
    buttons.append(button)

# variables
hangman_status = 0

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# setup
FPS = 60
clock = pygame.time.Clock()
run = True

letter_cord = {}
for val in list(zip(buttons, letters)):
    letter_cord[val[0]] = val[1]


def draw():
    window.fill(WHITE)
    for k, v in letter_cord.items():
        x = v[0]
        y = v[1]
        window.blit(k, (x, y))
    window.blit(images[hangman_status], (150, 100))
    pygame.display.update()


while run:
    clock.tick(FPS)
    draw()
    for event in pygame.event.get():
        # quit event
        if event.type == pygame.QUIT:
            run = False
        # click coord
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            print(pos)


pygame.quit()
