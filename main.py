# rename the folder to pong AI #
import neat
import pygame
import pygame_gui
import time
import random
import sys

from pygame.locals import *


def game(genomes, config):
  nets = []
  opponets = []
  for id, g in genomes:
    net = neat.nn.FeedForwardNetwork.create(g, config)
    nets.append(net)
  # ---------------------------------------------- #
  pygame.init()

  window_size = (800, 600)
  display = pygame.display.set_mode(window_size)
  pygame.display.set_caption('pong Ai')
  
  clock = pygame.time.Clock()

  ball = pygame.Rect(350, 350, 15, 15)
  player = pygame.Rect(10, 350, 10, 50)
  opponet = pygame.Rect(window_size[0] - 20, 350, 10, 50)

  ball_color = (100, 100, 255)
  player_color = (200, 150, 10)
  opponet_color = (200, 150, 90)

  opponets.append(opponet)

  paddles_speed = 5
  ball_speed = [4 * random.choice([1, -1]), 4 * random.choice([1, -1])]

  score = 0
  score_font = pygame.font.Font(pygame.font.get_default_font(), 25)

  opponet_score = 0

  moving_up = False
  moving_down = False

  while True:
    display.fill((25, 25, 25))

    pygame.draw.line(display, (200, 200, 200), (window_size[0]/2, 0), (window_size[0]/2, 700))
    
    value = score_font.render(f'{score} : {opponet_score}', True, (255, 255, 255))
    display.blit(value, [window_size[0]/2-25, 20])

    pygame.draw.rect(display, player_color, player)
    pygame.draw.rect(display, opponet_color, opponet)
    pygame.draw.ellipse(display, (90,90,120), ball)

    ball.x += ball_speed[0]
    ball.y += ball_speed[1]

    # ai to smash you :)
    for index, opponet in enumerate(opponet):
      output = nets[index].activate(opponet.y, ball.x, ball.y)
      i = output.index(max(output))
      if i == 0:
        opponet.y -= paddles_speed
      else:
        opponet.y += paddles_speed
    # bouncing off walls and some scoring
    if ball.top <= 0 or ball.bottom >= window_size[1]:
      ball_speed[1] *= -1
    
    if ball.left <= 0:
      ball.x = window_size[0]/2
      ball.y = window_size[1]/2
      ball_speed = [4 * random.choice([1, -1]), 4 * random.choice([1, -1])]
      opponet_score += 1
    if ball.right >= window_size[0]:
      ball.x = window_size[0]/2
      ball.y = window_size[1]/2
      ball_speed = [4 * random.choice([1, -1]), 4 * random.choice([1, -1])]
      score += 1
    # collide with player
    if ball.colliderect(player) or ball.colliderect(opponet):
      ball_speed[0] *= -1
    # forcing the player & the AI to stay on the screen
    if player.top <= 0:
      player.top = 0
    if player.bottom >= window_size[1]:
      player.bottom = window_size[1]
    if opponet.top <= 0:
      opponet.top = 0
    if opponet.bottom >= window_size[1]:
      opponet.bottom = window_size[1]


    if moving_up:
      player.y -= paddles_speed
    elif moving_down:
      player.y += paddles_speed
    else:
      pass

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
      
      if event.type == pygame.KEYDOWN:
        if event.key in [K_w, K_UP]:
          moving_up = True
        if event.key in [K_s, K_DOWN]:
          moving_down = True
      
      if event.type == KEYUP:
        if event.key in [K_w, K_UP]:
          moving_up = False
        if event.key in [K_s, K_DOWN]:
          moving_down = False
    
    pygame.display.update()
    clock.tick(60)




if __name__ == "__main__":
  config_path = "./config.txt"
  config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

  p = neat.Population(config)

  p.add_reporter(neat.StdOutReporter(True))
  stats = neat.StatisticsReporter()
  p.add_reporter(stats)


  p.run(game, 50)
