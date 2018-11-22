import numpy as np

def augment(x, y, max_left_padding=1, max_top_padding=1):
  rx = []
  ry = []
  left_padding=[]
  top_padding=[]
  
  for i in range(max_left_padding):
    for j in range(max_top_padding):
      rx.append(x)
      ry.append(y)
      left_padding.append(i),
      top_padding.append(j)
  return rx,ry,left_padding,top_padding



# print(augment('x', 1))
# print(augment('x', 1, max_left_padding=2))
# print(augment('x', 1, max_top_padding=2))
# print(augment('x', 1, max_left_padding=2, max_top_padding=3))