
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


def color_create(classes):
    colors = []
    for i in range(len(classes)):
        color = (tuple(np.random.choice(range(256), size=3)))
        colors.append((int(color[0]),int(color[1]),int(color[2])))
        
    return colors

def read_image(path):
    return cv2.imread(path)

def prepare_image(image):
    
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    #image = normalize(to_tensor(im_pil))
    image = to_tensor(im_pil)
    
    image = image.to(device)
    return image

def obj_detector(model, image, thresh = 0.5):
     # Forward prop.
    predicteds = model(image.unsqueeze(0))
    bboxes, scores, labels = predicteds[0]["boxes"], predicteds[0]["scores"], predicteds[0]["labels"]
    num = torch.argwhere(scores > thresh).shape[0]
    
    return bboxes.cpu().data, scores.cpu().data, labels.cpu().data , num


def predicted(bboxes, scores, labels, num, image,class_name,colors, show = False, draw = False):
   
    num = torch.argwhere(scores > 0.9).shape[0] # to do predict
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype('int')
        
        if draw:
            color_ = colors[int(labels.numpy()[i])]
            if int(labels.numpy()[i]) < 0:
                color_ = colors[0]
            current_class = class_name[labels.numpy()[i] - 1]

            image = cv2.rectangle(image,(x1, y1),(x2, y2),color = color_ , thickness =1)
            
            image = cv2.putText(image,
                           text = str(current_class),
                           org = (x1, y1),
                           fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale = 1,
                           color = color_)
    
    if(show):
        plt.imshow(image)
        plt.show()
        
    return image




