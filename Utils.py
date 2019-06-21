class Seed(Enum):
  B = 1
  F = 2
  
class Region(Enum):
  SF = 10
  WF = 80
  WB = 160
  SB = 240

def load_data(path):
  masks  = {}
  images = {}
  for l in tqdm(list(path.glob('*.png'))):
    id = l.parts[-1].split('.')[0]
    mask = cv.imread(str(l))
    masks[int(id)] = mask
    
  for l in tqdm(list(path.glob('*.jpg'))):
    id = l.parts[-1].split('.')[0]
    image = cv.imread(str(l))
    images[int(id)] = image
    
  ids = sorted(list(images.keys()))
  images_, masks_ = [], []
  for id in ids:
    images_.append(images[id])
    masks_.append(masks[id])
    
  return images_, masks_

def resize_data(images, masks, new_size=(84,84)):
  for i, (image, mask) in enumerate(zip(images, masks)):
    images[i] = cv.resize(image, new_size)
    masks[i]  = cv.resize(mask, new_size)
    
  return images, masks



def create_seed_mask(center, value=Seed.B, diameter=3, size=(84, 84)):
  '''
  Takes coordinates and type of seed point, diameter and image size and returns an
  image of the same size containing the value (Seed.F or Seed.B) in the pixels
  forming the seed points
  
  params
  ---------
  center:    (row, col) tuple
             Center pixel of the seed points
  
  value:     {Seed.F, Seed.B} enum
             Type of the seed point either F (Foreground) / B (Background)
  
  diameter:  int
             Diameter of the seed point in pixels (3 at train time / 13 at test time)
  
  size:      tuple
             Size of the resulting mask
  '''
  value = value.value
  grid = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
  sum_grid = (grid[0] - center[1]) ** 2 + (grid[1] - center[0]) ** 2
  idxs = np.where((sum_grid) <= (diameter/2)**2)
  seed_image = np.zeros_like(image)
  seed_image[idxs] = value
  return seed_image

def merge_seed_masks(masks):
  final_result = np.zeros_like(masks[0])
  for mask in masks:
    final_result += mask
  final_result[final_result >= 3] = 2
  
  return final_result

def calculate_IoU(mask1, mask2):
  if len(mask1.shape) > 2:
    mask1 = mask1[:,:,0]
    
  if len(mask2.shape) > 2:
    mask2 = mask2[:,:,0]
  
  idxs1 = np.where(mask1 > 0)
  idxs2 = np.where(mask2 > 0)
  
  idxs1 = set([(x, y) for x, y in zip(idxs1[0], idxs1[1])])
  idxs2 = set([(x, y) for x, y in zip(idxs2[0], idxs2[1])])
  
  intersection = len(idxs1.intersection(idxs2))
  union        = len(idxs1.union(idxs2))
  
  return intersection/union

def get_image_different_regions(mask, kernel_size, debug=True):
  kernel = np.ones(kernel_size ,np.uint8)
  eroded  = cv.erode(mask, kernel)
  dilated = cv.dilate(mask, kernel)
  
  SF_region = eroded
  WF_region = cv.bitwise_and(mask, cv.bitwise_not(eroded))
  WB_region = cv.bitwise_and(dilated, cv.bitwise_not(mask))
  SB_region = cv.bitwise_not(dilated)
  
  all_regions = np.zeros_like(mask)

  all_regions[np.where(WF_region>0)] = Region.WF.value
  all_regions[np.where(SF_region>0)] = Region.SF.value
  all_regions[np.where(WB_region > 0)] = Region.WB.value
  all_regions[np.where(SB_region > 0)] = Region.SB.value

  return all_regions

def get_all_different_regions(masks, kernel_size=(5,5)):
  if len(masks.shape) > 3:
    masks = masks[:,:,:,0]
    
  all_masks_with_regions = masks.copy()
  for i in range(masks.shape[0]):
    mask = masks[i]
    all_masks_with_regions[i] = get_image_different_regions(mask, kernel_size=kernel_size)

  return all_masks_with_regions

  def get_random_pixels(idxs, count):
  count_idxs = idxs[0].shape[0]
  index_indices = np.arange(count_idxs)
  if count_idxs < count:
    index_indices = np.random.choice(index_indices, count)
  chosen_indices = np.random.permutation(index_indices)[:count]
  chosen_pixels  = np.array([idxs[0][chosen_indices], idxs[1][chosen_indices]])
  return chosen_pixels.T
  
def generate_initial_seeds(all_masks_with_regions, num_sets=5):
  initial_foreground_seeds = np.empty((10000, num_sets, 2))
  initial_background_seeds = np.empty((10000, num_sets, 2))
  for i in range(all_masks_with_regions.shape[0]):
    mask_with_regns = all_masks_with_regions[i]
    strong_foreground_region = np.where(mask_with_regns == Region.SF.value)
    if strong_foreground_region[0].shape[0] == 0:
      strong_foreground_region = np.where(mask_with_regns == Region.WF.value)
    initial_foreground_seeds[i] = get_random_pixels(strong_foreground_region, num_sets)
    
    strong_background_region = np.where(mask_with_regns == Region.SB.value)
    initial_background_seeds[i] = get_random_pixels(strong_background_region, num_sets)
    
  return initial_foreground_seeds, initial_background_seeds