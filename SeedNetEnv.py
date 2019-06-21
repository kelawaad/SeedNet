from Utils import *

class SeedNetEnv:
    def __init__(self, 
                path='./MSRA_Dataset/MSRA10K_Imgs_GT/Imgs/'):
        self.path = Path(path)
        self.images, self.masks = load_data(self.path)
        self.images, self.masks = resize_data(self.images, self.masks)
        self.images = np.array(self.images)
        self.masks  = np.array(self.masks)[:,:,:,0]
        self.masks[self.masks>0] = 255
        self.train_images, self.val_images = self.images[:9000],self.images[9000:] 
        self.train_masks, self.val_masks = self.masks[:9000],self.masks[9000:] 
        self.image  = None
        self.gt_mask = None
        self.gt_mask_with_regions = None
        self.history = None
        self.all_masks_with_regions = get_all_different_regions(self.masks, kernel_size=(5,5))
        self.train_masks_with_regions = self.all_masks_with_regions[:9000]
        self.val_masks_with_regions   = self.all_masks_with_regions[9000:]
        self.initial_foreground_seeds, self.initial_background_seeds = generate_initial_seeds(self.all_masks_with_regions)
        self.train_initial_foreground_seeds = self.initial_foreground_seeds[:9000]
        self.val_initial_foreground_seeds = self.initial_foreground_seeds[9000:]
        self.train_initial_background_seeds = self.initial_background_seeds[:9000]
        self.val_initial_background_seeds = self.initial_background_seeds[9000:]
        print(self.images.shape, self.masks.shape)
        print(f'\nEnvironment initialized.\nThe data contains {len(self.images)} images')    
    
    def seed(self, seed=1337):
        np.random.seed(seed)
        print(f'Numpy seed with {seed}')
    
    def step(self, action):
        u = (action % 400) / 20
        u = round(u * (84/20.0))
        v = (action % 400) %  20
        v = round(v * (84/20.0))
        print(u,v)

        seed_layer = Seed.F if action < 400 else Seed.B   
        mask = create_seed_mask((u, v), value=seed_layer)
        self.history.append(mask)
        final_mask = merge_seed_masks(self.history)
        pred_mask = random_walker(self.image, final_mask, beta=150, multichannel=True)
        pred_mask[pred_mask!=Seed.F.value] = 0
        reward = calculate_reward(self.gt_mask_with_regions, (u,v), seed_layer, self.gt_mask, pred_mask, k=5)

        # print(f'reward_exp: {reward}')
        # print(f'img: {self.image.shape}')
        # print(f'pred_mask: {pred_mask.shape}')
        # print(f'history lenght: {len(self.history)}')
        # print(f'u: {u} , v: {v}')

        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # axs[0][0].imshow(self.image)
        # axs[0][0].set_title('Original Image')
        # axs[0][1].imshow(self.gt_mask)
        # axs[0][1].set_title('Ground Truth Mask')
        # axs[1][0].imshow(self.image)
        # axs[1][0].imshow(pred_mask*100, alpha=0.8)
        # axs[1][0].imshow(final_mask*100,alpha=0.4)
        # axs[1][0].set_title('Generated Mask')
        # axs[1][1].imshow(final_mask*100)
        # axs[1][1].imshow(self.gt_mask, alpha=0.4, cmap='gray')
        # axs[1][1].set_title('Seeds used for mask generation using random walker')

        state = np.empty((84,84,4))
        state[:,:,:3] = self.image
        state[:,:,-1] = pred_mask

        done = True if len(self.history)==12 else False

        return reward, state, done


    def reset(self, img_idx=654, state='train'):
        if state == 'train':
            self.image  = self.train_images[img_idx]
            self.gt_mask_with_regions = self.train_masks_with_regions[img_idx]
            self.gt_mask = self.train_masks[img_idx]
            self.history = []
        
            foreground_seed = self.train_initial_foreground_seeds[img_idx][0]
            foreground_mask = create_seed_mask((foreground_seed[0], foreground_seed[1]), value=Seed.F)
            self.history.append(foreground_mask)

            background_seed = self.train_initial_background_seeds[img_idx][0]
            background_mask = create_seed_mask((background_seed[0], background_seed[1]), value=Seed.B)
            self.history.append(background_mask)
        elif state == 'validation':
            self.image  = self.val_images[img_idx]
            self.gt_mask_with_regions = self.val_masks_with_regions[img_idx]
            self.gt_mask = self.val_masks[img_idx]
            self.history = []
        
            foreground_seed = self.val_initial_foreground_seeds[img_idx][0]
            foreground_mask = create_seed_mask((foreground_seed[0], foreground_seed[1]), value=Seed.F)
            self.history.append(foreground_mask)

            background_seed = self.val_initial_background_seeds[img_idx][0]
            background_mask = create_seed_mask((background_seed[0], background_seed[1]), value=Seed.B)
            self.history.append(background_mask)

        print("Environment Reset Successfully.")
        plt.imshow(self.image)
        plt.grid(False)
        plt.title("Current Image\n("+ state +" Set)")

