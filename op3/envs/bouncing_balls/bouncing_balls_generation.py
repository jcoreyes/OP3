import numpy as np
import matplotlib
import h5py
from argparse import ArgumentParser
import op3.envs.blocks.mujoco.utils.data_generation_utils as dgu


#########Start util functions#########
def np_one_hot(an_array, depth):
    the_size = an_array.shape[0]
    tmp = np.zeros((the_size, depth))
    tmp[np.arange(the_size), an_array] = 1
    return tmp
#########End util functions#########


##########Start classes for individual types of physical entities#########
class VerticalWall(object):
    def __init__(self, startCoord, endCoord, vel=np.zeros(2)):
        self.startC = np.array(startCoord) * 1.0
        self.endC = np.array(endCoord) * 1.0
        self.entityType = "vwall"
        self.mass = float("inf")
        self.v = vel

    def collide(self, aBall):
        if abs(self.startC[0] - aBall.p[0]) < aBall.r:
            if (self.startC[1] > aBall.p[1] and self.endC[1] < aBall.p[1]) or \
                    (self.startC[1] < aBall.p[1] and self.endC[1] > aBall.p[1]):
                if self.startC[0] - aBall.p[0] > 0:  # If ball collided from right side
                    aBall.v[0] = -abs(aBall.v[0]) - self.v[0]
                else:
                    aBall.v[0] = abs(aBall.v[0]) + self.v[0]
                return True
        return False

    def moveVel(self, vel, factor=1):
        self.startC += vel * factor
        self.endC += vel * factor

    def move(self, factor=1):
        self.startC += self.v * factor
        self.endC += self.v * factor


class HorizontalWall(object):
    def __init__(self, startCoord, endCoord, vel=np.zeros(2)):
        self.startC = np.array(startCoord) * 1.0
        self.endC = np.array(endCoord) * 1.0
        self.entityType = "hwall"
        self.mass = float("inf")
        self.v = vel

    def collide(self, aBall):
        if abs(self.startC[1] - aBall.p[1]) < aBall.r:
            if (self.startC[0] > aBall.p[0] and self.endC[0] < aBall.p[0]) or \
                    (self.startC[0] < aBall.p[0] and self.endC[0] > aBall.p[0]):
                if self.startC[1] - aBall.p[1] > 0:  # If ball collided from bottom
                    aBall.v[1] = -abs(aBall.v[1]) - self.v[1]
                else:
                    aBall.v[1] = abs(aBall.v[1]) + self.v[1]
                return True
        return False

    def moveVel(self, vel, factor=1):
        self.startC += vel * factor
        self.endC += vel * factor

    def move(self, factor=1):
        self.startC += self.v * factor
        self.endC += self.v * factor


class Rectangle(object):
    def __init__(self, botLeftX, botLeftY, topRightX, topRightY, vel, color, fillColor):
        self.bottomWall = HorizontalWall([botLeftX, botLeftY], [topRightX, botLeftY])
        self.topWall = HorizontalWall([botLeftX, topRightY], [topRightX, topRightY])
        self.leftWall = VerticalWall([botLeftX, botLeftY], [botLeftX, topRightY])
        self.rightWall = VerticalWall([topRightX, botLeftY], [topRightX, topRightY])

        self.v = vel
        self.allWalls = [self.bottomWall, self.topWall, self.leftWall, self.rightWall]
        self.entityType = "rectangle"
        self.color = color
        self.fillColor = fillColor

        self.blc = np.array([botLeftX, botLeftY]) * 1.0
        self.height = topRightY - botLeftY
        self.width = topRightX - botLeftX
        self.area = self.height * self.width

    def move(self, factor=1):
        self.blc += self.v * factor
        for aWall in self.allWalls:
            aWall.moveVel(self.v, factor)

    def moveVel(self, aVel):
        self.blc += aVel
        for aWall in self.allWalls:
            aWall.moveVel(aVel, 1)

    def collide(self, anEntity):
        if anEntity.entityType == "ball":
            collisionOccurred = False
            for aWall in self.allWalls:
                tmp = aWall.collide(anEntity)
                collisionOccurred = tmp or collisionOccurred
            return collisionOccurred
        elif anEntity.entityType == "rectangle":
            if self.intersect(anEntity):
                self.v = -self.v
                anEntity.v = -anEntity.v
                return True
            return False
        else:
            print("HERERERER")
            raise Exception("Unknown entity type in collide")

    def inside(self, aPos, aRad):
        if not self.fillColor:  # If not self.fillColor, this brick can contain objects in it and should therefore return False
            return False

        if aPos[0] > self.blc[0] - aRad and aPos[0] < self.blc[0] + self.width + aRad \
                and aPos[1] > self.blc[1] - aRad and aPos[1] < self.blc[1] + self.height + aRad:
            return True
        return False

    def intersect(self, otherRect):
        if not self.fillColor and not otherRect.fillColor:
            return False

        tmp1, tmp2 = np.maximum(self.blc, otherRect.blc)  # tmp1 = max of blc[0]'s
        tmp3 = min(self.blc[0] + self.width, otherRect.blc[0] + otherRect.width)
        tmp4 = min(self.blc[1] + self.height, otherRect.blc[1] + otherRect.height)
        areaI = (tmp3 - tmp1) * (tmp4 - tmp2) + 0.001  # In case of precision errors

        if (tmp3 - tmp1) < 0 and (tmp4 - tmp2) < 0:
            return False

        if not otherRect.fillColor:
            return areaI < self.area
        if not self.fillColor:
            return areaI < otherRect.area
        if areaI > 0:
            return True
        return False

    def matricize(self, xyMatrix):
        if not self.fillColor:
            return np.zeros((xyMatrix.shape[0], xyMatrix.shape[1])), False

        frame = np.array([[self.inside(xy, 0) for xy in innerMat] for innerMat in xyMatrix])  # Boolean mask
        frame = frame.astype(np.float32)  # W*H
        frame = np.stack([frame, frame, frame], -1)
        frame[frame[:, :, i] == True, ...] = self.color
        return frame, True


class Ball(object):
    def __init__(self, radius, mass, pos, vel=np.zeros(2), color=1):
        self.r = radius * 0.5
        self.m = mass
        self.p = pos
        self.v = vel
        self.color = color

        self.entityType = "ball"

    def move(self, factor=1):
        self.p += self.v * factor

    def collide(self, anotherBall):
        if np.linalg.norm(self.p - anotherBall.p) < self.r + anotherBall.r:
            deltaV = self.v - anotherBall.v
            deltaP = self.p - anotherBall.p
            deltaV1 = 2 * anotherBall.m / (self.m + anotherBall.m) * \
                      np.dot(deltaV, deltaP) / (np.linalg.norm(deltaP) ** 2) * \
                      deltaP
            deltaV2 = 2 * self.m / (self.m + anotherBall.m) * \
                      np.dot(deltaV, deltaP) / (np.linalg.norm(deltaP) ** 2) * \
                      -deltaP
            self.v = self.v - deltaV1
            anotherBall.v = anotherBall.v - deltaV2

            u = 0.5 * (self.r + anotherBall.r - np.linalg.norm(deltaP))

            self.p += u * deltaP / np.linalg.norm(deltaP)
            anotherBall.p += -u * deltaP / np.linalg.norm(deltaP)
            return True
        return False

    def matricize(self, xyMatrix):
        frame = np.linalg.norm(xyMatrix - self.p, axis=-1) < self.r
        frame = frame.astype(np.float32)
        frame = np.stack([frame, frame, frame], -1)
        frame[frame[:, :, i] == True, ...] = self.color  # int(self.color * 255.0)
        return frame, True


class Agent(object):
    def __init__(self, originalEntity, otherUnmovableEntities):
        self.ent = originalEntity
        self.otherUnmovableEnts = otherUnmovableEntities
        self.entityType = self.ent.entityType

    def moveVel(self, velDir):  # Moves the agent, if it collides with an unmovable entity then go back to last location
        self.ent.moveVel(velDir)
        for anEntity in self.otherUnmovableEnts:
            if self.ent.collide(anEntity):
                self.ent.moveVel(velDir * -1)
                return

    def collide(self, anE):
        return self.ent.collide(anE)

    def inside(self, aPos, aRad):
        return self.ent.inside(aPos, aRad)

    def matricize(self, xyMatrix):
        return self.ent.matricize(xyMatrix)
##########End classes for individual types of physical entities#########


##########Start helpful plotting function##########
def plotEntities(listOfEntinties, ballFade):
    for anE in listOfEntinties:
        if anE.entityType == "ball":
            plt.scatter([anE.p[0]], [anE.p[1]], color=str(ballFade))
            ax.add_artist(plt.Circle((anE.p[0], anE.p[1]), anE.r, color='b', fill=False))
        elif anE.entityType == "rectangle":
            rect = matplotlib.patches.Rectangle(anE.blc, anE.width, anE.height, fill=anE.fillColor,
                                                edgecolor='r', linewidth=1, alpha=0.1)
            ax.add_patch(rect)

##########End helpful plotting function##########


###########Start functions for creating a single simulationa############
def ar(start, end, res):
    return np.arange(start, end, (end - start) / res, dtype='float') + (end - start) / res / 2


def randomBrick(image_size, minBrickSize, maxBrickSize, colors):
    width, height = np.random.randint(minBrickSize, maxBrickSize, size=2)
    blX = np.random.random_sample() * (2 * image_size - width - 0.0001) - image_size
    blY = np.random.random_sample() * (2 * image_size - height - 0.0001) - image_size
    a_color = pickRandomColor(colors)
    theBrick = Rectangle(blX, blY, blX + width, blY + height, np.array([0.5, 0]), a_color, True)
    return theBrick


def randomFixedPosition(image_size, brickInfos, colors):
    bricks = []
    for aBrickInfo in brickInfos:
        isBad = True
        while isBad:
            newBrick = randomBrick(image_size, aBrickInfo[0], aBrickInfo[1], colors)
            isBad = False
            for pastBrick in bricks:
                if pastBrick.intersect(newBrick):
                    isBad = True
        bricks.append(newBrick)
    return bricks


def randomAddedBalls(image_size, balls, bricks):
    PADDING = 2
    paddedImageSize = image_size - 2
    goodConfig = False
    numBalls = len(balls)
    # Note: Possible infinite loop if bricks are sized in such a way that no valid configuration exists
    while not goodConfig:
        goodConfig = True
        p = np.random.uniform(-paddedImageSize, paddedImageSize,
                              size=(numBalls, 2))
        for i in range(numBalls):
            for aBrick in bricks:
                if aBrick.inside(p[i], balls[i].r):
                    goodConfig = False

        for i in range(numBalls):
            for j in range(i):
                if np.linalg.norm(p[i] - p[j]) < balls[i].r + balls[j].r:
                    goodConfig = False

    for i in range(numBalls):
        balls[i].p = p[i]


def matricizeFrame(allEntities, size, res):
    [I, J] = np.meshgrid(ar(-size, size, res), ar(-size, size, res))

    xyMatrix = np.dstack((I, J))

    curFrame = np.zeros((res, res, 3))

    groupFrame = np.zeros((res, res, 1))
    displayedGroups = 0
    for i in range(len(allEntities)):
        frame, isDisplayed = allEntities[i].matricize(xyMatrix)
        if len(frame.shape) == 2:
            frame = np.stack([frame, frame, frame], -1)
        if isDisplayed:
            displayedGroups += 1
            curFrame = np.where(frame > 0, frame, curFrame)

            tmp1 = np.sum(frame, -1)
            tmp1[tmp1 != 0] = displayedGroups
            groupFrame = np.maximum(groupFrame, np.expand_dims(tmp1, -1))

    background_color = allEntities[0].color
    groupFrame = groupFrame.astype(np.float32)

    curFrame = np.where(curFrame > 1., 1., curFrame)
    curFrame = np.where(groupFrame < 1, background_color, curFrame)
    return curFrame, groupFrame


# Black, White, Red, Lime, Blue, Yellow, Cyan, Magenta
COLOR_LIST = [[1, 1, 1], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
              [255, 0, 255]]


def pickBackgroundColor(some_colors):
    if some_colors is None:
        return np.random.uniform(low=0.0, high=1.0, size=3), None
    tmp = np.random.randint(0, len(some_colors))
    bg_color = some_colors[tmp]
    other_colors = some_colors[:tmp] + some_colors[tmp + 1:]
    return np.array(bg_color) / 255, other_colors


def pickRandomColor(some_colors):
    if some_colors is None:
        return np.random.uniform(low=0.0, high=1.0, size=3)
    tmp = np.random.randint(0, len(some_colors))
    return np.array(some_colors[tmp]) / 255  # As rgb should be between 0 and 1


def createSingleSim(half_grid_size, res, masses, radii, brickInfos, colors, haveAgent=True, FRAMES=20,
                    TIMESTEPS_PER_FRAME=1):
    half_grid_size = half_grid_size * 1.0
    n = np.size(masses)
    bg_color, other_colors = pickBackgroundColor(colors)
    m = masses
    r = radii

    v = np.random.randn(n, 2)
    v = v / np.linalg.norm(v, axis=1).reshape(n, 1) * 0.5
    balls = [Ball(r[i], m[i], np.zeros(2), v[i], pickRandomColor(other_colors)) for i in range(n)]

    bricks = [Rectangle(-half_grid_size, -half_grid_size, half_grid_size, half_grid_size, np.zeros(2), bg_color,
                        False)]  # The playing field
    if brickInfos:  # If there are non-agent, non-outer boundary bricks
        bricks.extend(randomFixedPosition(half_grid_size, brickInfos, other_colors))
    randomAddedBalls(half_grid_size, balls, bricks)  # Modifies balls for a valid starting position

    allEntities = []

    actions_one_hot = None
    if haveAgent:  #Create the agent
        minMoveLen, maxMoveLen = 3, 15
        agent = Agent(bricks[-1], bricks[:-1])  # Agent is the last brick
        bricks = bricks[:-1]
        randActions = np.random.randint(-1, 2, FRAMES * TIMESTEPS_PER_FRAME)  # Actions are either -1, 0, or 1
        randActionLens = np.random.randint(minMoveLen, maxMoveLen, FRAMES) * TIMESTEPS_PER_FRAME
        actions = np.repeat(randActions, randActionLens)[:FRAMES * TIMESTEPS_PER_FRAME]
        actions_one_hot = np_one_hot(actions, 3)
        constVel = np.array([0.4, 0])

    allEntities.extend(bricks)
    allEntities.extend(balls)

    frames = []
    groupFrames = []
    isBad = False
    for t in range(FRAMES * TIMESTEPS_PER_FRAME):
        # Move all the entities
        [anE.move() for anE in allEntities]

        if haveAgent:
            for anE in allEntities:  # Handle collisions between agent and objects
                agent.collide(anE)
            # Move the agent
            agent.moveVel(actions[t] * constVel)

        for i in range(len(bricks)):
            for k in range(i + 1, len(bricks)):
                bricks[i].collide(bricks[k])
                # if bricks[i].collide(bricks[k]):
                #     print(t, i, k)

        # Handle balls hitting bricks
        for aBall in balls:
            for aBrick in bricks:
                aBrick.collide(aBall)

        # Handle balls hitting other balls
        for ballI in range(n):
            for ballJ in range(ballI):
                balls[ballI].collide(balls[ballJ])

        for aBall in balls:
            for aBrick in bricks:
                if aBrick.inside(aBall.p, -0.2):
                    isBad = True
                    return -1, -1, isBad
            if haveAgent and agent.inside(aBall.p, -0.2):
                isBad = True
                return -1, -1, isBad

        if haveAgent:
            for anE in allEntities:
                agent.collide(anE)

        if t % TIMESTEPS_PER_FRAME == 0:
            if haveAgent:
                curFrame, curGroupFrame = matricizeFrame(allEntities + [agent], half_grid_size, res)
            else:
                curFrame, curGroupFrame = matricizeFrame(allEntities, half_grid_size, res)
            frames.append(curFrame)
            groupFrames.append(curGroupFrame)
    return np.array(frames), np.array(groupFrames), actions_one_hot, isBad
###########End functions for creating a single simulation############


###########Start function for creating a dataset of multiple simulations############
def createMultipleSims(filename, datasets, n_frames, n, m, m_to_r, n_of_bricks, brick_sizes, colors,
                       have_agent, half_grid_size, image_res):
    with h5py.File(filename, 'w') as f:
        for folder in datasets:
            cur_folder = f.create_group(folder)

            num_sims = datasets[folder]

            # create datasets, write to disk
            image_data_shape = (n_frames, num_sims, image_res, image_res, 3)
            groups_data_shape = (n_frames, num_sims, image_res, image_res, 1)
            features_dataset = cur_folder.create_dataset('features', image_data_shape, dtype='float32')
            groups_dataset = cur_folder.create_dataset('groups', groups_data_shape, dtype='float32')

            if have_agent:
                action_data_shape = (n_frames, num_sims, 3)
                action_dataset = cur_folder.create_dataset('actions', action_data_shape, dtype='float32')

            i = 0
            while i < num_sims:
                num_balls = np.random.choice(n)
                num_bricks = np.random.choice(n_of_bricks)
                ball_ms = np.random.choice(m, num_balls)
                ball_rs = [m_to_r[an_m] for an_m in ball_ms]

                sim_brick_info = [brick_sizes] * num_bricks  # Multiplying a list duplicates it
                frames, group_frames, actions_one_hot, isBad = createSingleSim(half_grid_size, image_res,
                                                                               ball_ms, ball_rs, sim_brick_info, colors,
                                                                               have_agent, n_frames)

                if isBad:
                    continue

                if have_agent:
                    action_dataset[:, i, :] = actions_one_hot

                frames = np.expand_dims(frames, 1)
                group_frames = np.expand_dims(group_frames, 1)

                features_dataset[:, [i], :, :, :] = frames
                groups_dataset[:, [i], :, :, :] = group_frames
                i += 1

            print("Done with dataset: {}".format(folder))


###########End function for creating a dataset of multiple simulations############


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None)        # Filename
    parser.add_argument('-n', '--num_balls', type=str, default='3')        # Range of number of balls
    parser.add_argument('-m', '--masses', type=str, default='1')           # Range of ball masses
    parser.add_argument('-r', '--radii', type=str, default='3')            # Range of ball radii
    parser.add_argument('-i', '--image_res', type=int, default=64)         # Image size
    parser.add_argument('-nb', '--num_bricks', type=str, default='0')      # Range of bricks
    parser.add_argument('-bs', '--brick_sizes', type=str, default='36')    # Range of brick sizes
    parser.add_argument('-nf', '--num_frames', type=int, default=21)       # Number of frames per sequence
    parser.add_argument('-hgs', '--half_grid_size', type=int, default=10)  # Size of playing field
    parser.add_argument('-ns', '--num_sims', type=int, default=1000)       # Number of sims to run
    parser.add_argument('-mi', '--make_images', type=bool, default=False)  # Visualize the first 10 images
    parser.add_argument('-ag', '--agent', type=bool, default=False)        # Create a very simply brick agent
    parser.add_argument('-c', '--color', type=int, default=0)              # Number of colors to choose, if 0 then continuous colors

    args = parser.parse_args()
    print(args)

    if args.filename is None:
        args.filename = 'n' + args.num_balls + 'nb' + args.num_bricks + 'm' + args.masses + 'r' + args.radii + \
                        'bs' + args.brick_sizes + '.h5'  # Default filename
    if args.filename[-3:] != ".h5":
        args.filename += ".h5"
    print("Saving dataset to file: {}".format(args.filename))

    datasets = {'training': args.num_sims, 'validation': min(args.num_sims, 1000)}
    num_balls = [int(aChar) for aChar in args.num_balls]
    m = [int(aChar) for aChar in args.masses]
    r = [int(aChar) for aChar in args.radii]
    m_to_r = {}
    for i in range(len(m)):
        m_to_r[m[i]] = r[i]

    num_bricks = [int(aChar) for aChar in args.num_bricks]
    brick_sizes = [int(aChar) for aChar in args.brick_sizes]
    assert len(brick_sizes) == 2

    if args.color == 0:
        colors = None
    else:
        colors = COLOR_LIST[:args.color]

    createMultipleSims(args.filename, datasets, args.num_frames, num_balls, m, m_to_r,
                       num_bricks, brick_sizes, colors, args.agent, args.half_grid_size, args.image_res)

    if args.make_images:
        print("Done with creating the dataset, now creating visuals")
        dgu.hdf5_to_image(args.filename)
        for i in range(10):
            dgu.make_gif("imgs/training/{}/features".format(str(i)), "animation.gif")
            dgu.make_gif("imgs/training/{}/groups".format(str(i)), "animation.gif")





