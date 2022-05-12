import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import time 
import cv2

# count how many images in each folder(based on how many balls in hand)
def count_img():
  # root_folder = './refine_train'
  root_folder = './candidate'

  dirs= os.listdir(root_folder)
  print(dirs)
  to=0
  ot=0
  for dir in dirs:
    direct=os.path.join(root_folder,dir)
    to=len([name for name in os.listdir(direct) if os.path.isfile(os.path.join(direct, name))])
    ot+=to
    print(direct,to)
  print(ot)

# distribute images to balance dataset
def rename():
  direct=os.path.join('./refine_train')
  new_dirct =os.path.join('./refine_validation')
  folders=os.listdir(direct)
  for folder in folders:
    count = os.path.join(direct,folder)
    new_path = os.path.join(new_dirct,folder)
    images=os.listdir(count)
    # print(images[0])
    total = 68

    for image in images:
      image_path = os.path.join(count,image)
      if os.path.isfile(image_path):  
          print(new_path,image)
          new_img_path = os.path.join(new_path,image)
          os.rename(image_path,new_img_path)
          total -=1
      if total ==0:
        break

# read from npy file and concatenate them
def create_fused():
  root_img_path = './raw_npy_data/40_mm_sphere_images_'
  root_label_path = './raw_npy_data//40_mm_sphere_data_'
  x_box = [30, 110]
  y_box = [20, 100]
  folder_path = './fused_raw'
  total_count = 5000
  iter = 0
  # count = np.ones(10, dtype=int) * 70000
  # count = [4303,3391,2330,1704,203,13]
  # count = [10003,10001,10000,10004,10003,1003]
  count = [30003,30001,30000,30004,30003,3003]

  for i in range(12,15):
      img_path = root_img_path+ str(i) + '.npy'
      label_path = root_label_path + str(i) + '.npy'
      sim = np.load(img_path)
      sim_label = np.load(label_path)
      print(f'{sim.shape[0]}\n{sim_label.shape}, {i}')

      for idx in range(sim.shape[0]):
          ball_count = int(sim_label[idx, 0])
          if ball_count >=5:
            ball_count = 5
          # if count[ball_count]>limit[ball_count]:
          #   continue
          
          current_folder = os.path.join(folder_path, str(ball_count))
          if not os.path.isdir(current_folder):
              os.mkdir(current_folder)

          fig = plt.figure()
          fig.set_size_inches(3.8, 3.8)

          plt.subplot(2, 2, 1)
          plt.imshow(sim[idx, 0, x_box[0]:x_box[1], y_box[0]:y_box[1]])
          plt.axis('off')
          plt.subplot(2, 2, 2)
          plt.imshow(sim[idx, 1, x_box[0]:x_box[1], y_box[0]:y_box[1]])
          plt.axis('off')
          plt.subplot(2, 2, 3)
          plt.imshow(sim[idx, 2, x_box[0]:x_box[1], y_box[0]:y_box[1]])
          plt.axis('off')
          plt.subplot(2, 2, 4)
          plt.imshow(sim[idx, 3, x_box[0]:x_box[1], y_box[0]:y_box[1]])
          plt.axis('off')

          plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
          img_name = '{:0>4d}.png'.format(count[ball_count])
          # plot_img_np = get_img_from_fig(fig,dpi=100)
          # print(plot_img_np.shape)
          count[ball_count] += 1
          merged_img_path = os.path.join(current_folder, img_name)
          # img = cv2.resize(fig,(380,380))
          # cv2.imwrite(img_path,img)
          plt.savefig(merged_img_path, dpi=100)
          fig.clear()
          plt.close()
      #     break
      # break
      # os.remove(img_path)
      del sim
      gc.collect()

# add small noise(white dots) to the images
def noisy(image):
  w,col,ch = image.shape
  s_vs_p = 0.5
  amount = 0.04
  out = np.copy(image)
  # Salt mode
  num_salt = np.ceil(amount * image.size * s_vs_p)
  coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
  out[coords] = 1

  # Pepper mode
  num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
  coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
  out[coords] = 0
  return out


# increase the number of images in desire folder
def augmentation():
  
  direct=os.path.join('./candidate')
  folders=os.listdir(direct)
  for folder in folders:
    if folder == '4' or folder =='5':
      count = os.path.join(direct,folder)
      images=os.listdir(count)
      # cur = len([name for name in os.listdir(count) if os.path.isfile(os.path.join(count, name))])
      more_image = 1500 - len(images)
      print(more_image)
      while more_image != 0:
        idx = np.random.randint(len(images))
        img_name = os.path.join(count,images[idx])
        image = cv2.imread(img_name)
        aug_type = np.random.randint(2)

        new_img_name = time.time()
        if aug_type == 0:
        # add random noise:
          image = noisy(image)
        elif aug_type == 1:
          # random rotation
          rand_angle = np.random.randint(3,size=1)
        if rand_angle == 0:
          image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
        elif  rand_angle == 1:
          image = cv2.rotate(image, cv2.ROTATE_180)
        else: 
          image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        filename = os.path.join(count,str(new_img_name)+'.png')
        print(filename)
        cv2.imwrite(filename, image)
        # plt.imshow(image)
        # plt.show
        more_image -= 1

count_img()
# rename()