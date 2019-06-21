from Utils import *

split_size = 9000

class SeedNetEnv:
    def __init__(self, 
                path='./MSRA_Dataset/MSRA10K_Imgs_GT/Imgs/'):
        self.path = Path(path)
        self.images, self.masks = load_data(self.path)
        self.images, self.masks = resize_data(self.images, self.masks)
        self.images = np.array(self.images)
        self.masks  = np.array(self.masks)[:,:,:,0]
        self.masks[self.masks>0] = 255
        self.train_images, self.val_images = self.images[:split_size],self.images[split_size:] 
        self.train_masks, self.val_masks = self.masks[:split_size],self.masks[split_size:] 
        self.image  = None
        self.gt_mask = None
        self.gt_mask_with_regions = None
        self.history = None
        self.all_masks_with_regions = get_all_different_regions(self.masks, kernel_size=(5,5))
        self.train_masks_with_regions = self.all_masks_with_regions[:split_size]
        self.val_masks_with_regions   = self.all_masks_with_regions[split_size:]
        self.initial_foreground_seeds, self.initial_background_seeds = generate_initial_seeds(self.all_masks_with_regions)
        self.train_initial_foreground_seeds = self.initial_foreground_seeds[:split_size]
        self.val_initial_foreground_seeds = self.initial_foreground_seeds[split_size:]
        self.train_initial_background_seeds = self.initial_background_seeds[:split_size]
        self.val_initial_background_seeds = self.initial_background_seeds[split_size:]
        print(self.images.shape, self.masks.shape)
        print(f'\nEnvironment initialized.\nThe data contains {len(self.images)} images')    
    
    def seed(self, seed=1337):
        np.random.seed(seed)
        print(f'Numpy seed with {seed}')
    
    def step(self, action):
        u = (action % 400) / 20
        u = round(u * (83/20.0))
        v = (action % 400) %  20
        v = round(v * (83/20.0))
        # print(u,v)

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
        state[:,:,3]  = pred_mask
        # print(state.shape)
        state = np.swapaxes(state, 0, 2)

        done = True if len(self.history)==12 else False

        return state, reward, done


    def reset(self, state='train'):
        if state == 'train':
            img_idx = np.random.randint(0,split_size)
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
            img_idx = np.random.randint(0,1000)
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
        
        # mask = create_seed_mask((u, v), value=seed_layer)
        # self.history.append(mask)
        final_mask = merge_seed_masks(self.history)
        pred_mask = random_walker(self.image, final_mask, beta=150, multichannel=True)
        pred_mask[pred_mask!=Seed.F.value] = 0
    
        state = np.empty((84,84,4))
        state[:,:,:3] = self.image
        state[:,:,3]  = pred_mask
        state = np.swapaxes(state, 0, 2)
        # print("Environment Reset Successfully.")
        # plt.imshow(self.image)
        # plt.grid(False)
        # plt.title("Current Image\n("+ state +" Set)")

        return state