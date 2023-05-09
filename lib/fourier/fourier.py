import torch
import numpy as np


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[..., 0] ** 2 + fft_im[..., 1] ** 2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2(fft_im[..., 1], fft_im[..., 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
    amp_src[..., 0:b, 0:b] = amp_trg[..., 0:b, 0:b]  # top left
    amp_src[..., 0:b, w - b:w] = amp_trg[..., 0:b, w - b:w]  # top right
    amp_src[..., h - b:h, 0:b] = amp_trg[..., h - b:h, 0:b]  # bottom left
    amp_src[..., h - b:h, w - b:w] = amp_trg[..., h - b:h, w - b:w]  # bottom right
    return amp_src

def low_freq_mutate_features(amp_src, amp_trg, L=0.1):
    _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b
    amp_src[..., 0:b, 0:b] = amp_trg[..., 0:b, 0:b]  # top left
    amp_src[..., 0:b, w - b:w] = amp_trg[..., 0:b, w - b:w]  # top right
    amp_src[..., h - b:h, 0:b] = amp_trg[..., h - b:h, 0:b]  # bottom left
    amp_src[..., h - b:h, w - b:w] = amp_trg[..., h - b:h, w - b:w]  # bottom right
    return amp_src

#
def features_source_to_target(src_features, trg_features, L=0.1):
    # exchange magnitude
    # input: src_features, trg_features

    # get fft of both source and target
    fft_src = torch.fft.fft2(src_features.clone(), dim=(-2, -1))
    fft_trg = torch.fft.fft2(trg_features.clone(), dim=(-2, -1))

    fft_src_2dim = torch.stack((fft_src.real, fft_src.imag), -1)
    fft_trg_2dim = torch.stack((fft_trg.real, fft_trg.imag), -1)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src_2dim.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg_2dim.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate_features(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src_2dim.size(), dtype=torch.float)
    fft_src_[..., 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[..., 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    # _, feature_num = src_features.size()
    src_in_trg = torch.fft.ifft2(torch.complex(fft_src_[..., 0], fft_src_[..., 1]), dim=(-2, -1)).real

    return src_in_trg


def img_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.fft.fft2(src_img.clone(), dim=(-2, -1))
    fft_trg = torch.fft.fft2(trg_img.clone(), dim=(-2, -1))

    fft_src_2dim = torch.stack((fft_src.real, fft_src.imag), -1)
    fft_trg_2dim = torch.stack((fft_trg.real, fft_trg.imag), -1)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src_2dim.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg_2dim.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src_2dim.size(), dtype=torch.float)
    fft_src_[..., 0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[..., 1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.ifft2(torch.complex(fft_src_[..., 0], fft_src_[..., 1]), dim=(-2, -1)).real

    return src_in_trg
