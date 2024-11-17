import numpy as np
import pygame
import random
from collections import deque

# Pygame setup
pygame.init()

# Step 1: Environment Creation
def create_environment():
    """
    Create a 2D grid environment with obstacles, empty spaces, and a target.
    0 - Empty space, 1 - Obstacle, 'T' - Target, 'A' - Agent
    """
    env = np.zeros((10, 10), dtype=object)  # Create a 10x10 grid of empty spaces (0)

    # Add 20 obstacles, ensuring that they are not adjacent to each other
    obstacles = 0
    while obstacles < 20:
        # Randomly choose a location for an obstacle
        row = random.randint(0, env.shape[0] - 1)
        col = random.randint(0, env.shape[1] - 1)

        # Ensure the location is not occupied by the agent, target, or another obstacle
        # Also, ensure no adjacent obstacles (check surrounding cells)
        if env[row, col] == 0:
            if not any(env[row + dr, col + dc] == 1
                       for dr in [-1, 0, 1] if 0 <= row + dr < env.shape[0]
                       for dc in [-1, 0, 1] if 0 <= col + dc < env.shape[1]):
                # Add the obstacle
                env[row, col] = 1
                obstacles += 1

    # Set the target ('T')
    env[0, 9] = 'T'  # Place target at top-right corner

    # Set the agent ('A')
    env[9, 0] = 'A'  # Place the agent at bottom-left corner

    return env


# Step 2: Agent Definition
class Agent:
    def __init__(self, start_position, environment):
        self.position = start_position  # Agent's current position (row, col)
        self.environment = environment  # Environment (grid world)

    def sense_environment(self):
        """
        Sense the surrounding environment (adjacent cells).
        Returns a list of valid neighboring positions.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        neighbors = []

        for direction in directions:
            new_row = self.position[0] + direction[0]
            new_col = self.position[1] + direction[1]

            # Check if the neighbor is within bounds and is not an obstacle
            if 0 <= new_row < self.environment.shape[0] and 0 <= new_col < self.environment.shape[1]:
                if self.environment[new_row, new_col] != 1:  # Not an obstacle
                    neighbors.append((new_row, new_col))

        return neighbors

    def move(self, new_position):
        """ Move the agent to a new position """
        self.position = new_position


# BFS Pathfinding Algorithm (Replacing A*)
def bfs(start, target, env):
    """
    Perform BFS to find the shortest path from start to target in the environment.

    Parameters:
    - start: starting position (row, col)
    - target: target position (row, col)
    - env: the environment (2D grid)

    Returns:
    - path: a list of positions [(row, col)] representing the shortest path
    """
    # Initialize the queue and the visited set
    queue = deque([(start, [start])])  # Stores (current_pos, path_so_far)
    visited = set([start])

    while queue:
        current_pos, path_so_far = queue.popleft()

        # If we reach the target, return the path
        if current_pos == target:
            return path_so_far

        # Get neighbors
        for neighbor in Agent(current_pos, env).sense_environment():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path_so_far + [neighbor]))

    return []  # Return empty path if no path is found


# Visualization Helper Function using Pygame
def plot_environment(env, agent_position, target_position):
    """
    Plot the environment grid, showing the agent, target, and obstacles using pygame.
    """
    screen.fill((255, 255, 255))  # Fill the screen with white color

    # Define grid cell size
    cell_size = 50

    # Loop through the environment and draw it
    for row in range(env.shape[0]):
        for col in range(env.shape[1]):
            x = col * cell_size
            y = row * cell_size

            if env[row, col] == 1:  # Obstacle
                pygame.draw.rect(screen, (0, 128, 0), (x, y, cell_size, cell_size))  # Dark Green for obstacles
            elif env[row, col] == 'T':  # Target
                pygame.draw.rect(screen, (255, 0, 0), (x, y, cell_size, cell_size))  # Red for target
            elif env[row, col] == 'A':  # Agent
                pygame.draw.rect(screen, (0, 0, 255), (x, y, cell_size, cell_size))  # Blue for agent
            else:  # Path (empty space)
                pygame.draw.rect(screen, (144, 238, 144), (x, y, cell_size, cell_size))  # Light Green for paths

            pygame.draw.rect(screen, (200, 200, 200), (x, y, cell_size, cell_size), 1)  # Grid lines

    pygame.display.flip()  # Update the display


# Main simulation
def main():
    # Set up pygame window
    global screen
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Agent Pathfinding')

    # Create the environment
    env = create_environment()

    # Set up agent starting position and target position
    agent_start = (9, 0)  # Agent starts at bottom-left corner
    target_pos = (0, 9)  # Target is at top-right corner

    # Initialize the agent
    agent = Agent(agent_start, env)

    # Find the path using BFS
    path = bfs(agent_start, target_pos, env)

    if not path:
        print("No path found to the target!")
        return

    # Simulate agent finding the target
    for step in path:
        # Clear the previous agent position in the environment
        env[agent_start[0], agent_start[1]] = 0

        # Set the new agent position in the environment
        env[step[0], step[1]] = 'A'
        agent_start = step

        # Plot the environment with the agent's new position
        plot_environment(env, agent_start, target_pos)

        # Pause for better visualization
        pygame.time.delay(500)

    # Wait until the user closes the window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


# Run the simulation
if __name__ == '__main__':
    main()
