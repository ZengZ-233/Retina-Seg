from DataSet import *
import numpy as np
from pre_processing import my_PreProc



def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the image
    img_w = full_imgs.shape[3]  # width of the image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim
    if (leftover_h != 0):  # change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " + str(stride_h))
        # print("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        print("So the H dim will be padded with additional " + str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):  # change dimension of img_w
        print("the side W is not compatible with the selected stride of " + str(stride_w))
        # print("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
        print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        print("So the W dim will be padded with additional " + str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new padded images shape: " + str(full_imgs.shape))
    return full_imgs


def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * (
                (img_w - patch_w) // stride_w + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    print("Number of patches on h : " + str(((img_h - patch_h) // stride_h + 1)))
    print("Number of patches on w : " + str(((img_w - patch_w) // stride_w + 1)))
    print("number of patches per image: " + str(N_patches_img) + ", totally for testset: " + str(N_patches_tot))
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


def get_data_test_overlap(test_data_path_list, patch_height, patch_width, stride_height, stride_width):
    test_imgs_original, test_masks = load_data(test_data_path_list)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.
    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)
    print("\nTest images shape: {}, vaule range ({} - {}):" \
          .format(test_imgs.shape, str(np.min(test_imgs)), str(np.max(test_imgs))))

    # extract the test patches from the all test images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    print("test patches shape: {}, value range ({} - {})" \
          .format(patches_imgs_test.shape, str(np.min(patches_imgs_test)), str(np.max(patches_imgs_test))))

    return patches_imgs_test, test_imgs_original, test_masks, test_imgs.shape[2], test_imgs.shape[3]


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    # print("N_patches_h: " + str(N_patches_h))
    # print("N_patches_w: " + str(N_patches_w))
    # print("N_patches_img: " + str(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    # N_full_imgs = preds.shape[0]
    print("There are " +str(N_full_imgs) +" images in Testset")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))
    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k] # Accumulate predicted values
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1  # Accumulate the number of predictions
                k+=1
    # assert(k==preds.shape[0])
    # assert(np.min(full_sum)>=1.0)
    final_avg = full_prob/full_sum # Take the average
    # print(final_avg.shape)
    # assert(np.max(final_avg)<=1.0) # max value for a pixel is 1.0
    # assert(np.min(final_avg)>=0.0) # min value for a pixel is 0.0
    return final_avg


def pred_only_in_FOV(data_imgs, data_masks):
    assert (len(data_imgs.shape) == 4 and len(data_masks.shape) == 4)  # 4D arrays
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  # loop over the all test images
        for x in range(width):
            for y in range(height):
                new_pred_imgs.append(data_imgs[i, :, y, x])
                new_pred_masks.append(data_masks[i, :, y, x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

