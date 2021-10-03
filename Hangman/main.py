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

# variables
hangman_status = 0

# colors
WHITE = (255, 255, 255)

# setup
FPS = 60
clock = pygame.time.Clock()
run = True

while run:
    clock.tick(FPS)
    window.fill(WHITE)
    window.blit(images[hangman_status], (150, 100))
    pygame.display.update()
    for event in pygame.event.get():
        # quit event
        if event.type == pygame.QUIT:
            run = False
        # click coord
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()


pygame.quit()
