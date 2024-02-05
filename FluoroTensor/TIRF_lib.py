# Copyright (C) 2023: Max F. K. Wills & Ian C. Eperon

# FluoroTensor: identification and tracking of colocalised molecules and their stoichiometries in multi-colour single molecule imaging via deep learning
# Max F.K. Wills, Carlos Bueno Alejo, Nikolas Hundt, Andrew J. Hudson, Ian C. Eperon
# bioRxiv 2023.11.21.567874; doi: https://doi.org/10.1101/2023.11.21.567874

# LICENCE:
# This program is made available under the Creative Commons NonCommercial 4.0 licence.


#      Disclaimer of Warranties and Limitation of Liability.
#
#   a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
#      EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
#      AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
#      ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
#      IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
#      WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
#      PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
#      ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
#      KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
#      ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.
#
#   b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
#      TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
#      NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
#      INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
#      COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
#      USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
#      ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
#      DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
#      IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.
#
#   c. The disclaimer of warranties and limitation of liability provided
#      above shall be interpreted in a manner that, to the extent
#      possible, most closely approximates an absolute disclaimer and
#      waiver of all liability.

# This notice applies to the following code unless specified to be exempt (see scipy cookbook functions at the end)


import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy import optimize
import cv2
# from skimage import restoration, morphology

from _TIRF_lib  import moments, fitgaussian2D, gaussian2D, moments2, fitgaussian2D2, gaussian2D2

def gauss(x_size, y_size, x0, y0, sigma_x, sigma_y, A, mode="depr", maximum=2):
    gaussian = np.zeros((y_size, x_size, 3))
    for x in range(x_size):
        for y in range(y_size):
            num1 = (x - x0)**2
            num2 = (y - y0)**2
            den1 = 2 * (sigma_x**2)
            den2 = 2 * (sigma_y**2)
            if mode == "depr":
                gaussian[y][x][0] = maximum - (A*np.exp(-(num1/den1 + num2/den2)))
                gaussian[y][x][1] = gaussian[y][x][0]
                gaussian[y][x][2] = gaussian[y][x][0]
            elif mode == "protr":
                gaussian[y][x][0] = A*np.exp(-(num1/den1 + num2/den2))
    return gaussian


def gauss2(x_size, y_size, x0, y0, sigma_x, sigma_y, A, z_off):
    gaussian = np.zeros((y_size, x_size))
    for x in range(x_size):
        for y in range(y_size):
            num1 = (x - x0)**2
            num2 = (y - y0)**2
            den1 = 2 * (sigma_x**2)
            den2 = 2 * (sigma_y**2)
            gaussian[y][x] = (A*np.exp(-(num1/den1 + num2/den2))) + z_off
    return gaussian


""" build large kernel matrix and normalize when this library is imported"""
large_filter = gauss(129, 129, 64, 64, 32, 32, 1, mode="protr")[:, :, 0]
large_filter = large_filter / np.sum(large_filter)
conv_filter = gauss(5, 5, 2, 2, 1, 1, 1, mode="protr")[:, :, 0]
conv_max = np.sum(conv_filter)
conv_filter = conv_filter / conv_max

convolution_kernel = gauss(5, 5, 2, 2, 1, 1, 1, mode="protr")[:, :, 0]
convolution_kernel_slim = gauss(5, 5, 2, 2, 0.6, 0.6, 1, mode="protr")[:, :, 0]
conv_max = np.sum(convolution_kernel)
convolution_kernel = convolution_kernel / conv_max
conv_max = np.sum(convolution_kernel_slim)
convolution_kernel_slim = convolution_kernel_slim / conv_max

wavelet_kernel = gauss(7, 7, 3, 3, 1.8, 1.8, 1, mode="protr")[:, :, 0]

# footprint = morphology.disk(3)

phased_kernel = np.array([[-1, -1, -1, -1, -1],
                          [-1, 1.7, 1.7, 1.7, -1],
                          [-1, 1.7, 2, 1.7, -1],
                          [-1, 1.7, 1.7, 1.7, -1],
                          [-1, -1, -1, -1, -1]])

defocus_kernel = np.zeros((9, 19))

dfk1 = gauss(14, 9, 3, 4.5, 1.6, 1.6, 1, mode="protr")[:, :, 0]
dfk2 = gauss(14, 9, 10, 3.5, 1.6, 1.6, 1, mode="protr")[:, :, 0]
defocus_kernel = dfk1 + dfk2
dfk_max = np.sum(defocus_kernel)
defocus_kernel = defocus_kernel / dfk_max
# plt.imshow(defocus_kernel)
# plt.show()

deconv_subtract = 0.009
deconvolution_wavelet = defocus_kernel - deconv_subtract
deconv_integral = np.sum(deconvolution_wavelet)
print(f"\nDeconvolution Wavelet Integral: {deconv_integral}\n")


def load_tiff(path):

    if path:
        raw_tif = Image.open(path)
        # print("TIF file shape:", np.shape(raw_tif))
        h, w = np.shape(raw_tif)
        tif_array = np.zeros((h, w, raw_tif.n_frames), dtype='float32')
        for index in range(raw_tif.n_frames):
            raw_tif.seek(index)
            tif_array[:, :, index] = np.array(raw_tif)
        final_tif_array = tif_array.astype(np.single)
        # print(final_tif_array.shape)
        length = raw_tif.n_frames
        return final_tif_array, length


# def top_hat(tif_img, multiplier):
#     # background = restoration.rolling_ball(tif_img)
#     # plt.imshow(background)
#     # plt.show()
#     # result = tif_img - background
#
#     footprint = morphology.disk(3)
#     final = morphology.white_tophat(tif_img, footprint)
#     maximum = np.max(final)
#     final = (final / maximum) * 255
#
#     """ this hist is needed for cummulative thresholding, do not comment out! """
#     int_bin = np.bincount(np.ravel(np.copy(final).astype(int)))
#     hist = []
#     for b in range(len(int_bin)):
#             hist.append(int_bin[b])
#
#     cumulative = []
#     accumulator = 0
#     for i in range(len(hist)):
#         accumulator += hist[i]
#         cumulative.append(accumulator)
#
#     hist_cumulative = np.array(cumulative)
#     percentile = multiplier
#     inequality_iterator = hist_cumulative > np.max(hist_cumulative) * percentile
#     truncated = hist_cumulative[inequality_iterator]
#
#     threshold = 255 - len(truncated)
#
#     final = final - threshold
#     final = cv2.filter2D(final, -1, convolution_kernel_slim)
#     mx = np.max(final)
#     final = (final / mx) * 255
#     final = final * 2
#     final = np.clip(final, 0, 255)
#
#     return final, hist


def wavelet_transform(tif_image, multiplier, subtraction, flags):
    wavelet_kernel_final = wavelet_kernel - subtraction
    wavelet_sum = np.sum(wavelet_kernel_final)
    # print(f"Wavelet Integral: {wavelet_sum}")
    enhanced = cv2.filter2D(tif_image, -1, wavelet_kernel_final)
    maximum = np.max(enhanced)
    enhanced = np.clip(enhanced, 0, maximum)
    maximum = np.max(enhanced)
    enhanced = (enhanced / maximum) * 255

    truncate_point = 255 - int((multiplier * 255))
    enhanced = np.clip(enhanced, 0, truncate_point)
    enhanced = np.clip(enhanced * (255 / truncate_point), 0, 255)
    enhanced = cv2.filter2D(enhanced, -1, convolution_kernel_slim)
    int_bin = np.bincount(np.ravel(np.copy(enhanced).astype(int)))
    hist = []
    for b in range(len(int_bin)):
        hist.append(int_bin[b])
    hist[0] = 0

    if flags:
        deconvolved = deconvolution(enhanced, 0.8)
        return deconvolved, hist

    return enhanced, hist


def deconvolution(array, multiplier):
    array = array + 255
    array = array / 2
    # plt.imshow(array)
    # plt.show()
    enhanced = cv2.filter2D(array, -1, deconvolution_wavelet)
    # plt.imshow(enhanced)
    # plt.show()
    maximum = np.max(enhanced)
    enhanced = np.clip(enhanced, 0, maximum)
    # plt.imshow(enhanced)
    # plt.show()
    maximum = np.max(enhanced)
    enhanced = (enhanced / maximum) * 255
    truncate_point = 255 - int((multiplier * 255))
    enhanced = np.clip(enhanced, 0, truncate_point)
    enhanced = np.clip(enhanced * (255 / truncate_point), 0, 255)
    enhanced = cv2.filter2D(enhanced, -1, convolution_kernel_slim)
    return enhanced


def low_pass(tif_img, powr):
    conv = cv2.filter2D(tif_img, -1, large_filter, borderType=2)
    m = conv.max()
    conv = conv**powr
    conv = conv / conv.max()
    conv = conv * m
    low_pass_result = tif_img - conv
    low_pass_result = cv2.filter2D(low_pass_result, -1, convolution_kernel_slim, borderType=2)
    return low_pass_result


def true_low_pass(tif_img):
    conv = cv2.filter2D(tif_img, -1, large_filter, borderType=2)
    m = conv.max()
    conv = conv / m
    conv = conv * 255
    return conv


def create_sum(tif_img, frame_start, num_frames, view_mode):
    convolution_kernel = gauss(5, 5, 2, 2, 1, 1, 1, mode="protr")[:, :, 0]
    conv_max = np.sum(convolution_kernel)
    convolution_kernel = convolution_kernel / conv_max

    h, w = np.shape(tif_img)[0], np.shape(tif_img)[1]
    array = np.zeros((h, w))
    for frame in range(frame_start, frame_start + num_frames):
        array[:, :] = array[:, :] + tif_img[:, :, frame]

    max_val = array.max()
    array = (array / max_val) * 255

    tif_limit = tif_img[:, :, frame_start:frame_start+num_frames]
    max_index_matrix = np.argmax(tif_limit, axis=2)
    max_view = np.zeros((h, w))
    for x in range(w):
        for y in range(h):
            max_view[y, x] = tif_limit[y, x, max_index_matrix[y, x]]

    max_val = max_view.max()
    max_view = (max_view / max_val) * 255
    max_view = cv2.filter2D(max_view, -1, convolution_kernel_slim)

    print(view_mode)

    if view_mode == "s":
        return array

    array = array + max_view

    if view_mode == "m":
        return max_view

    max_val = array.max()
    array = (array / max_val) * 255

    if view_mode == "b":
        return array

    return array


def calculate_power(array, conv_num, multiplier, target):

    h, w = np.shape(array)[0], np.shape(array)[1]

    conv = cv2.filter2D(array, -1, convolution_kernel)
    for c_pass in range(conv_num):
        conv = cv2.filter2D(conv, -1, convolution_kernel)
    max_val = conv.max()
    conv = (conv / max_val) * 255

    convolution_residual = np.abs(conv - array)

    convolution_residual = cv2.filter2D(convolution_residual, -1, convolution_kernel_slim)
    max_val = convolution_residual.max()
    convolution_residual = (convolution_residual / max_val) * 255

    # hist = np.zeros((256))
    # for x in range(256):
    #     for y in range(256):
    #         hist[int(array[y][x])] += 1
    # plt.plot(hist, label="Original sum stack")
    # mean_org = np.mean(array)

    convolution_residual = 255 - convolution_residual

    convolution_residual = np.abs(convolution_residual)

    # hist = np.zeros((256))
    # for x in range(256):
    #     for y in range(256):
    #         hist[int(convolution_residual[y][x])] += 1
    # plt.plot(hist, label="Inverted convolution residual")
    # mean_inv = np.mean(convolution_residual)

    mx = np.max(convolution_residual)
    normal_residual = convolution_residual / mx

    mean = np.sum(normal_residual) / (h * w)
    power = np.log(0.6) / np.log(mean)

    # print(mean, pow)

    convolution_residual = np.abs(convolution_residual)
    convolution_residual = convolution_residual ** power
    max_val = convolution_residual.max()
    convolution_residual = (convolution_residual / max_val) * 255

    # hist = np.zeros((256))
    # for x in range(256):
    #     for y in range(256):
    #         hist[int(convolution_residual[y][x])] += 1
    # plt.plot(hist, label="Power scaled inverted convolution residual")
    # mean_invsc = np.mean(convolution_residual)
    # plt.axvline(mean_org, color="blue", linestyle="--", linewidth=1)
    # plt.axvline(mean_inv, color="orange", linestyle="--", linewidth=1)
    # plt.axvline(mean_invsc, color="green", linestyle="--", linewidth=1)
    # plt.legend()
    # plt.show()

    final = array * convolution_residual
    max_val = final.max()
    final = (final / max_val) * 255

    hist = np.zeros((256))
    for x in range(256):
        for y in range(256):
            hist[int(final[y][x])] += 1
    # plt.imshow(final)
    # plt.show()
    # plt.plot(hist)
    # plt.show()

    cumulative = []
    accumulator = 0
    for i in range(len(hist)):
        accumulator += hist[i]
        cumulative.append(accumulator)
    # plt.plot(cumulative)
    # plt.show()

    hist_cumulative = np.array(cumulative)
    percentile = multiplier
    inequality_iterator = hist_cumulative > np.max(hist_cumulative) * percentile
    truncated = hist_cumulative[inequality_iterator]
    # plt.plot(truncated)
    # plt.show()
    threshold = 255 - len(truncated)

    final = final - threshold
    mx = np.max(final)
    final = (final / mx) * 255
    final = final * 4
    final = np.clip(final, 0, 255)


    output = np.copy(final)

    sum_list = []

    for create in range(20, 200, 1):

        brightness = create / 20

        final = np.copy(output)

        final = final * brightness
        final = np.clip(final, 0, 255)

        ws = np.sum(final) / (h * w)
        sum_list.append(ws)

    resid = np.abs(np.array(sum_list) - target)
    ind = list(resid)
    index = ind.index(min(ind))

    brightness = (index+20) / 20

    return power, brightness


def sum_view(array, multiplier, bright, powr, conv_num):

    h, w = np.shape(array)[0], np.shape(array)[1]

    conv = cv2.filter2D(array, -1, convolution_kernel)
    for c_pass in range(conv_num):
        conv = cv2.filter2D(conv, -1, convolution_kernel)
    max_val = conv.max()
    conv = (conv / max_val) * 255

    # plt.imshow(array)
    # plt.show()
    # plt.imshow(conv)
    # plt.show()

    convolution_residual = conv - array
    max_val = np.max(convolution_residual)
    convolution_residual = np.clip(convolution_residual, 0, max_val)
    convolution_residual = cv2.filter2D(convolution_residual, -1, convolution_kernel_slim)
    max_val = convolution_residual.max()
    convolution_residual = (convolution_residual / max_val) * 255

    # plt.imshow(convolution_residual)
    # plt.show()

    convolution_residual = 255 - convolution_residual

    convolution_residual = np.abs(convolution_residual)

    # plt.imshow(convolution_residual)
    # plt.show()

    convolution_residual = convolution_residual ** powr
    max_val = convolution_residual.max()
    convolution_residual = (convolution_residual / max_val) * 255

    int_bin2 = np.bincount(np.ravel(np.copy(convolution_residual).astype(int)))
    hist2 = []
    for b in range(len(int_bin2)):
            hist2.append(int_bin2[b])
    # hist2 = np.zeros((256))
    # for x in range(256):
    #     for y in range(256):
    #         hist2[int(convolution_residual[y][x])] += 1

    # plt.plot(hist2)
    # plt.show()

    # plt.imshow(convolution_residual)
    # plt.show()

    final = array * convolution_residual

    max_val = final.max()
    final = (final / max_val) * 255
    final = np.clip(final, 0, 255)


    """ this hist is needed for cummulative thresholding, do not comment out! """
    int_bin = np.bincount(np.ravel(np.copy(final).astype(int)))
    hist = []
    for b in range(len(int_bin)):
            hist.append(int_bin[b])
    # hist = np.zeros((256))
    # for x in range(256):
    #     for y in range(256):
    #         hist[int(final[y][x])] += 1
    # plt.imshow(final)
    # plt.show()
    # plt.plot(hist)
    # plt.show()

    cumulative = []
    accumulator = 0
    for i in range(len(hist)):
        accumulator += hist[i]
        cumulative.append(accumulator)
    # plt.plot(cumulative)
    # plt.show()

    hist_cumulative = np.array(cumulative)
    percentile = multiplier
    inequality_iterator = hist_cumulative > np.max(hist_cumulative) * percentile
    truncated = hist_cumulative[inequality_iterator]

    # plt.plot(truncated)
    # plt.show()

    threshold = 255 - len(truncated)

    final = final - threshold
    mx = np.max(final)
    final = (final / mx) * 255
    final = final * 4
    final = np.clip(final, 0, 255)

    # plt.imshow(final)
    # plt.show()

    # integration = np.sum(final)
    # tot_average = integration / (h * w)
    # background = tot_average * multiplier
    # brightness = 255 / (255 - background) * bright
    #
    # final = final - background
    # final = final * brightness
    # final = np.clip(final, 0, 255)
    # final = final - background
    # final = final * brightness
    # final = np.clip(final, 0, 255)
    # final = final - background
    # final = final * brightness
    # final = np.clip(final, 0, 255)

    final = cv2.filter2D(final, -1, convolution_kernel_slim)
    # final_for_hist = np.copy(final).astype(int)
    # int_bin = np.bincount(np.ravel(final_for_hist))
    #
    # hist = []
    # for b in range(len(int_bin)):
    #         hist.append(int_bin[b])

    # plt.imshow(final)
    # plt.show()
                  # truncated
    return final, hist2


def create_image(stack, x, y):
    PIL_image = Image.fromarray(stack.astype('uint8'), 'RGB')
    PIL_image = PIL_image.resize((y, x), Image.NEAREST)
    return PIL_image


def detect_spots(view, img_array, criteria, mode):

    kernel = gauss(9, 9, 4, 4, 1.8, 1.8, 1.5, mode="protr")[:, :, 0] + (np.ones((9, 9)) * 0.0)
    kernel = normal_truncate(kernel)
    detection_field = np.copy(img_array)

    positions = []
    grid = (8, 8)
    stride = 8
    threshold = criteria["detection threshold"][view]
    averaging_dist = criteria["averaging distance"][view]
    gauss_fit_residual_threshold = criteria["minimum kernel residual"][view]

    min_sigma_threshold = criteria["minimum sigma"][view]
    max_sigma_threshold = criteria["maximum sigma"][view]
    min_intensity = criteria["minimum intensity"][view]
    min_amplitude = criteria["minimum gauss amplitude"][view]

    eccentricity_threshold = criteria["eccentricity threshold"][view]
    true_gauss_threshold = criteria["minimum gauss residual"][view]

    # scan an 8x8 grid across the image and list the coordinates of maximum points that have a peak / average
    # value greater than the threshold
    for x in range(0, len(detection_field) - grid[1] + 1, stride):
        for y in range(0, len(detection_field) - grid[0] + 1, stride):

            mini_field = detection_field[y:y+grid[1], x:x+grid[0]]
            maximum = np.max(mini_field)
            grid_sum = np.sum(mini_field) / (grid[0]*grid[1])
            loc_average = np.sum(mini_field) / (grid[0]*grid[1])
            global_mean = np.sum(detection_field) / (np.shape(detection_field)[0] * np.shape(detection_field)[1])
            if mode:
                limit = grid_sum
                limit_thresh = global_mean*threshold
            else:
                limit = maximum
                limit_thresh = global_mean*threshold

            if limit > limit_thresh:

                maxima_indices = np.where(mini_field == maximum)
                weighted_sum_X = 0
                weighted_sum_Y = 0

                for coord in range(len(maxima_indices[0])):
                    weighted_sum_X += maxima_indices[1][coord]
                    weighted_sum_Y += maxima_indices[0][coord]

                weighted_sum_X /= len(maxima_indices[0])
                weighted_sum_Y /= len(maxima_indices[0])
                positX = x + weighted_sum_X
                positY = y + weighted_sum_Y
                positions.append([positX, positY])

    # scan coordinate list and if maxima recorded in previous grids are within a threshold distance, as in the case
    # of the same spot spanning moe than one grid, average the positions together and save
    potential_spots = []

    for pos_index in range(len(positions)):
        position = positions[pos_index]
        averaged = False

        for spot_index in range(len(potential_spots)):
            compare = potential_spots[spot_index]
            dist = np.sqrt(abs(position[0]-compare[0])**2 + abs(position[1]-compare[1])**2)

            if dist < averaging_dist:
                position_sum = np.array(position) + np.array(compare)
                position_sum /= 2
                potential_spots[spot_index] = list(position_sum)
                averaged = True

        if not averaged:
            potential_spots.append(position)

    # scan through list of saved spots and create new single grids around potential spot locations to improve
    # positional accuracy for gaussian fitting
    spot_centres = []
    for spot in range(len(potential_spots)):
        mini_field = detection_field[int(potential_spots[spot][1]-4):int(potential_spots[spot][1]+4),
                                        int(potential_spots[spot][0]-4):int(potential_spots[spot][0]+4)]
        if len(mini_field) > 0:
            maximum = np.max(mini_field, initial=1)
            scale_factor = 255 / maximum
            mini_field2 = mini_field * scale_factor
            try:
                maxima_indices = np.where(mini_field2 > 100)
            except:
                maxima_indices = np.where(mini_field == maximum)
            weighted_sum_X = 0
            weighted_sum_Y = 0

            for coord in range(len(maxima_indices[0])):
                weighted_sum_X += maxima_indices[1][coord]
                weighted_sum_Y += maxima_indices[0][coord]

            if len(maxima_indices[0]) > 0:
                weighted_sum_X /= len(maxima_indices[0])
                weighted_sum_Y /= len(maxima_indices[0])
                positX = potential_spots[spot][0] + weighted_sum_X - 4
                positY = potential_spots[spot][1] + weighted_sum_Y - 4
                spot_centres.append([int(positX), int(positY)])

    # fit 9x9 gaussian kernel to 9x9 grid centred on potential spot and calculate residual. spot grid and kernel are
    # normalised so residual is independent of intensity. keep spot if residual below threshold. Then follow up by
    # fitting a 2D Gaussian to the spot
    spots = []

    for fit in range(len(spot_centres)):
        spot_grid = detection_field[int(spot_centres[fit][1]-4):int(spot_centres[fit][1]+5),
                                        int(spot_centres[fit][0]-4):int(spot_centres[fit][0]+5)]

        if np.shape(spot_grid) != (9, 9) and len(spot_grid) > 0:
            s_grid = spot_grid.ravel().copy()
            s_grid.resize((81))
            spot_grid = s_grid.reshape((9, 9))

        if len(spot_grid) > 0:
            gauss_fit_grid = np.copy(spot_grid)
            maximum = np.max(spot_grid)
            spot_grid = spot_grid / maximum
            residual = kernel - spot_grid
            residual = np.abs(residual)
            residual_sum = np.sum(residual)

            if residual_sum > gauss_fit_residual_threshold:
                """ Spot failed check, residual too high """
                pass
            else:
                if gauss_fit_grid.max() > min_intensity:

                    gauss_fit_grid = make_valid(gauss_fit_grid)
                    parameters = fitgaussian2D(gauss_fit_grid)
                    coordX = parameters[2] + spot_centres[fit][0] - 4
                    coordY = parameters[1] + spot_centres[fit][1] - 4

                    gauss_fit = gauss(9, 9, parameters[2], parameters[1], parameters[4], parameters[3],
                                      parameters[0] / maximum, mode="protr")[:, :, 0]
                    gauss_fit_resid = np.abs(gauss_fit - spot_grid)
                    gauss_fit_resid_sum = np.sum(gauss_fit_resid)
                    # print(gauss_fit_resid_sum)

                    if parameters[3] > min_sigma_threshold and parameters[4] > min_sigma_threshold and \
                    parameters[3] < max_sigma_threshold and parameters[4] < max_sigma_threshold and \
                    parameters[3] / parameters[4] > eccentricity_threshold and parameters[3] / parameters[4] < (1 / eccentricity_threshold) and \
                    parameters[0] > min_amplitude and gauss_fit_resid_sum <= true_gauss_threshold:

                        # spot data stored as [spotx, spoty, [sigmax, sigmay, amplitude, gaussian residual]]
                        spots.append([coordX, coordY, [parameters[4], parameters[3], parameters[0],
                                                       gauss_fit_resid_sum]])

    final_spot_list = [] 
    for check_dupe in range(len(spots)):
        reject = False
        for comparison in range(len(final_spot_list)):
            dist = np.sqrt(abs(spots[check_dupe][0] - final_spot_list[comparison][0]) ** 2 +
                           abs(spots[check_dupe][1] - final_spot_list[comparison][1]) ** 2)
            if dist < 3:
                reject = True
                continue
        if not reject:
            final_spot_list.append(spots[check_dupe])

    return final_spot_list


def continuous_track(view, img_array, criteria, mode, region=None, inverted=None):

    kernel = gauss(9, 9, 4, 4, 1.8, 1.8, 1.5, mode="protr")[:, :, 0] + (np.ones((9, 9)) * 0.0)
    kernel = normal_truncate(kernel)
    detection_field = np.copy(img_array)
    shape = np.shape(detection_field)
    h, w = shape[0], shape[1]
    enlarged = np.zeros((h+8, w+8))
    enlarged[4:-4, 4:-4] = detection_field
    detection_field = np.copy(enlarged)
    shape = np.shape(detection_field)
    h, w = shape[0], shape[1]

    positions = []
    grid = (8, 8)
    stride = 8
    threshold = criteria["detection threshold"][view]
    averaging_dist = criteria["averaging distance"][view]
    gauss_fit_residual_threshold = criteria["minimum kernel residual"][view]

    min_sigma_threshold = criteria["minimum sigma"][view]
    max_sigma_threshold = criteria["maximum sigma"][view]
    min_intensity = criteria["minimum intensity"][view]
    min_amplitude = criteria["minimum gauss amplitude"][view]

    eccentricity_threshold = criteria["eccentricity threshold"][view]
    true_gauss_threshold = criteria["minimum gauss residual"][view]

    # scan an 8x8 grid across the image and list the coordinates of maximum points that have a peak / average
    # value greater than the threshold

    region_min_x, region_min_y = 0, 0
    region_max_x, region_max_y = len(detection_field) - grid[1] + 1, len(detection_field) - grid[0] + 1

    if region is not None:
        region_min_x, region_min_y, region_max_x, region_max_y = region[0][0], region[0][1], region[1][0], region[1][1]
    if region is not None and inverted == 1:
        region_min_x, region_min_y, region_max_x, region_max_y = 0, 0, len(detection_field) - grid[1] + 1, len(detection_field) - grid[0] + 1


    for x in range(region_min_x, region_max_x, stride):
        for y in range(region_min_y, region_max_y, stride):
            if region is not None and inverted == 1:
                if x > region[0][0] and x < region[1][0] and y > region[0][1] and y < region[1][1]:
                    continue
            mini_field = detection_field[y:y+grid[1], x:x+grid[0]]
            if np.shape(mini_field)[0] == 0:
                mini_field = detection_field[y - 1:y + grid[1], x:x + grid[0]]
            if np.shape(mini_field)[1] == 0:
                mini_field = detection_field[y:y + grid[1], x - 1:x + grid[0]]
            if np.shape(mini_field)[0] == 0 and np.shape(mini_field)[1] == 0:
                mini_field = detection_field[y - 1:y + grid[1], x - 1:x + grid[0]]
            try:
                maximum = np.max(mini_field)
            except:
                maximum = 0
            grid_sum = np.sum(mini_field) / (grid[0]*grid[1])
            loc_average = np.sum(mini_field) / (grid[0]*grid[1])
            global_mean = np.sum(detection_field) / (np.shape(detection_field)[0] * np.shape(detection_field)[1])
            if mode:
                limit = grid_sum
                limit_thresh = global_mean*threshold
            else:
                limit = maximum
                limit_thresh = global_mean*threshold

            if limit > limit_thresh:

                maxima_indices = np.where(mini_field == maximum)
                weighted_sum_X = 0
                weighted_sum_Y = 0

                for coord in range(len(maxima_indices[0])):
                    weighted_sum_X += maxima_indices[1][coord]
                    weighted_sum_Y += maxima_indices[0][coord]

                weighted_sum_X /= len(maxima_indices[0])
                weighted_sum_Y /= len(maxima_indices[0])
                positX = x + weighted_sum_X
                positY = y + weighted_sum_Y
                positions.append([positX, positY])

    # scan coordinate list and if maxima recorded in previous grids are within a threshold distance, as in the case
    # of the same spot spanning moe than one grid, average the positions together and save
    potential_spots = []

    for pos_index in range(len(positions)):
        position = positions[pos_index]
        averaged = False

        for spot_index in range(len(potential_spots)):
            compare = potential_spots[spot_index]
            dist = np.sqrt(abs(position[0]-compare[0])**2 + abs(position[1]-compare[1])**2)

            if dist < averaging_dist:
                position_sum = np.array(position) + np.array(compare)
                position_sum /= 2
                potential_spots[spot_index] = list(position_sum)
                averaged = True

        if not averaged:
            potential_spots.append(position)

    # scan through list of saved spots and create new single grids around potential spot locations to improve
    # positional accuracy for gaussian fitting
    spot_centres = []
    for spot in range(len(potential_spots)):
        mini_field = detection_field[int(potential_spots[spot][1]-6):int(potential_spots[spot][1]+6),
                                        int(potential_spots[spot][0]-6):int(potential_spots[spot][0]+6)]
        if len(mini_field) > 0:
            maximum = np.max(mini_field, initial=1)
            maxima_indices = np.where(mini_field == maximum)
            weighted_sum_X = 0
            weighted_sum_Y = 0

            for coord in range(len(maxima_indices[0])):
                weighted_sum_X += maxima_indices[1][coord]
                weighted_sum_Y += maxima_indices[0][coord]

            if len(maxima_indices[0]) > 0:
                weighted_sum_X /= len(maxima_indices[0])
                weighted_sum_Y /= len(maxima_indices[0])
                positX = potential_spots[spot][0] + weighted_sum_X - 6
                positY = potential_spots[spot][1] + weighted_sum_Y - 6
                spot_centres.append([int(positX), int(positY)])

    # fit 9x9 gaussian kernel to 9x9 grid centred on potential spot and calculate residual. spot grid and kernel are
    # normalised so residual is independent of intensity. keep spot if residual below threshold. Then follow up by
    # fitting a 2D Gaussian to the spot
    spots = []

    for fit in range(len(spot_centres)):
        spot_grid = detection_field[int(spot_centres[fit][1]-4):int(spot_centres[fit][1]+5),
                                        int(spot_centres[fit][0]-4):int(spot_centres[fit][0]+5)]

        if np.shape(spot_grid) != (9, 9) and len(spot_grid) > 0:
            s_grid = spot_grid.ravel().copy()
            s_grid.resize((81))
            spot_grid = s_grid.reshape((9, 9))

        if len(spot_grid) > 0:
            gauss_fit_grid = np.copy(spot_grid)
            maximum = np.max(spot_grid, initial=1)
            spot_grid = spot_grid / maximum
            residual = kernel - spot_grid
            residual = np.abs(residual)
            residual_sum = np.sum(residual)

            if residual_sum > gauss_fit_residual_threshold:
                """ Spot failed check, residual too high """
                pass
            else:
                if gauss_fit_grid.max() > min_intensity:

                    gauss_fit_grid = make_valid(gauss_fit_grid)
                    parameters = fitgaussian2D(gauss_fit_grid)
                    coordX = parameters[2] + spot_centres[fit][0] - 8
                    coordY = parameters[1] + spot_centres[fit][1] - 8

                    gauss_fit = gauss(9, 9, parameters[2], parameters[1], parameters[4], parameters[3],
                                      parameters[0] / maximum, mode="protr")[:, :, 0]
                    gauss_fit_resid = np.abs(gauss_fit - spot_grid)
                    gauss_fit_resid_sum = np.sum(gauss_fit_resid)
                    # print(gauss_fit_resid_sum)

                    if parameters[3] > min_sigma_threshold and parameters[4] > min_sigma_threshold and \
                    parameters[3] < max_sigma_threshold and parameters[4] < max_sigma_threshold and \
                    parameters[3] / parameters[4] > eccentricity_threshold and parameters[3] / parameters[4] < (1 / eccentricity_threshold) and \
                    parameters[0] > min_amplitude and gauss_fit_resid_sum <= true_gauss_threshold:

                        # spot data stored as [spotx, spoty, [sigmax, sigmay, amplitude, gaussian residual]]
                        spots.append([coordX, coordY, [parameters[4], parameters[3], parameters[0],
                                                       gauss_fit_resid_sum]])

    final_spot_list = []
    for check_dupe in range(len(spots)):
        reject = False
        for comparison in range(len(final_spot_list)):
            dist = np.sqrt(abs(spots[check_dupe][0] - final_spot_list[comparison][0]) ** 2 +
                           abs(spots[check_dupe][1] - final_spot_list[comparison][1]) ** 2)
            if dist < 3:
                reject = True
                continue
        if not reject:
            final_spot_list.append(spots[check_dupe])

    return final_spot_list


def normal_truncate(matrix):
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            if matrix[y][x] > 1:
                matrix[y][x] = 1

    return matrix


def make_valid(grid):
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            try:
                test = int(grid[x][y])
            except ValueError:
                grid[x][y] = 0
    return grid


def add_spot(img_array, coords, mode):

    detection_field = np.copy(img_array)
    h, w = np.shape(detection_field)[0], np.shape(detection_field)[1]

    x_low = 4
    x_hi = 5
    y_low = 4
    y_hi = 5

    if coords[0] < 4:
        x_low = coords[0]
    if coords[0] > w - 6:
        x_hi = w - coords[0]
    if coords[1] < 4:
        y_low = coords[1]
    if coords[1] > h - 6:
        y_hi = h - coords[1]

    spot_grid = detection_field[int(coords[1] - y_low):int(coords[1] + y_hi), int(coords[0] - x_low):int(coords[0] + x_hi)]

    if not mode:
        detection_ratio = np.max(spot_grid) / (np.sum(detection_field) / (h*w))
    else:
        detection_ratio = (np.sum(spot_grid) / 64) / (np.sum(detection_field) / (h*w))

    parameters = fitgaussian2D(spot_grid)

    amp = parameters[0]
    cx, cy = coords[0] + parameters[2] - 4, coords[1] + parameters[1] - 4
    sigma_x = parameters[4]
    sigma_y = parameters[3]

    kernel = gauss(np.shape(spot_grid)[1], np.shape(spot_grid)[0], parameters[2], parameters[1],
                   sigma_x, sigma_y, amp, mode="protr")[:, :, 0]

    kmax = np.max(kernel)
    smax = np.max(spot_grid)
    kernel = kernel / kmax
    spot_grid = spot_grid / smax
    residual = np.abs(kernel - spot_grid)
    residual_sum = np.sum(residual)

    spot = [cx, cy, [sigma_x, sigma_y, amp, residual_sum, detection_ratio]]
    return spot


def track(img_array, cx, cy):
    try:
        detection_field = np.copy(img_array)
        h, w = np.shape(detection_field)[0], np.shape(detection_field)[1]
        x_low = 8
        x_hi = 9
        y_low = 8
        y_hi = 9
        if cx < 8:
            x_low = cx
        if cx > w - 10:
            x_hi = w - cx
        if cy < 8:
            y_low = cy
        if cy > h - 10:
            y_hi = h - cy
        spot_grid = detection_field[int(cy - y_low):int(cy + y_hi),
                    int(cx - x_low):int(cx + x_hi)]
        parameters = fitgaussian2D(spot_grid)
        amp = parameters[0]
        sx, sy = cx + parameters[2] - 8, cy + parameters[1] - 8
        sigma_x = parameters[4]
        sigma_y = parameters[3]
        kernel = gauss(np.shape(spot_grid)[1], np.shape(spot_grid)[0], parameters[2], parameters[1],
                       sigma_x, sigma_y, amp, mode="protr")[:, :, 0]
        kmax = np.max(kernel)
        smax = np.max(spot_grid)
        kernel = kernel / kmax
        spot_grid = spot_grid / smax
        residual = np.abs(kernel - spot_grid)
        residual_sum = np.sum(residual)

        spot = [sx, sy, [sigma_x, sigma_y, amp, residual_sum]]
        return spot
    except:
        return "Failed"


# The following functions are exempt from the above copyright notice licence
# They are made availbale under the following licence of the scipy cookbook:

# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2017 SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of Enthought nor the names of the SciPy Developers
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    Module from: https://scipy-cookbook.readthedocs.io/items/FittingData.html"""
    try:
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y
    except ValueError:
        return 1, 1, 1, 1, 1


def fitgaussian2D(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    Module from: https://scipy-cookbook.readthedocs.io/items/FittingData.html"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian2D(*p)(*np.indices(data.shape)) -
                                 data)
    try:
        p, success = optimize.leastsq(errorfunction, params)
        return p
    except TypeError:
        return None


def gaussian2D(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters
    Module from: https://scipy-cookbook.readthedocs.io/items/FittingData.html"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments2(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    Module from: https://scipy-cookbook.readthedocs.io/items/FittingData.html"""
    try:
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
        height = data.max()
        z_off = np.mean(data)
        return height, x, y, width_x, width_y, z_off
    except ValueError:
        return 1, 1, 1, 1, 1, 1


def fitgaussian2D2(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit
    Module from: https://scipy-cookbook.readthedocs.io/items/FittingData.html"""
    params = moments2(data)
    errorfunction = lambda p: np.ravel(gaussian2D2(*p)(*np.indices(data.shape)) -
                                 data)
    try:
        p, success = optimize.leastsq(errorfunction, params)
        return p
    except TypeError:
        return None


def gaussian2D2(height, center_x, center_y, width_x, width_y, z_off):
    """Returns a gaussian function with the given parameters
    Module from: https://scipy-cookbook.readthedocs.io/items/FittingData.html"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2) + z_off
