from matplotlib import pyplot as plt

def draw_img(path):
  with open('result.txt','r') as f:
    lines = f.readlines()
    result = lines[2:]

    img = []
    for line in result:
        line = line.split()
        line = list(map(float, line))
        img.append(line)

    plt.figure(figsize=[12,12])
    plt.imshow(img, cmap="hot")
    plt.colorbar()
draw_img('result.txt')