import numpy as np
import torch, math
from core.evaluate import accuracy
from fourier.fourier import features_source_to_target, img_source_to_target


class Combiner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        # epoch_number<90: div_epoch = epoch_number;
        # 90<=epoch_number<100 : div_epoch = 100;
        # 100<=epoch_number<=180: div_epoch = 200;
        # 180<epoch_number: div_epoch = epoch_number
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )

    def default(self, model, criterion, image, label, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        return loss, now_acc

    def bbn_mix(self, model, criterion, image, label, meta, **kwargs):

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
        # l = 0.5  # fix
        # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        # l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        # l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        # l = np.random.beta(self.alpha, self.alpha) # beta distribution
        # l = 1 if self.epoch <= 120 else 0  # seperated stage

        mixed_feature = 2 * torch.cat((l * feature_a, (1 - l) * feature_b), dim=1)
        output = model(mixed_feature, classifier_flag=True)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )

        return loss, now_acc

    def bbn_mix_features(self, model, criterion, image, label, meta, **kwargs):
        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )
        # feature_b -> feature_a domain
        feature_b = features_source_to_target(
            src_features=feature_b.view(-1, 8, 8),
            trg_features=feature_a.view(-1, 8, 8),
            L=self.cfg.FOURIER.LAMBDA)

        feature_b = feature_b.view(-1, 64).cuda()

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay
        # l = 0.5  # fix
        # l = math.cos((self.epoch-1) / self.div_epoch * math.pi /2)   # cosine decay
        # l = 1 - (1 - ((self.epoch - 1) / self.div_epoch) ** 2) * 1  # parabolic increment
        # l = 1 - (self.epoch-1) / self.div_epoch  # linear decay
        # l = np.random.beta(self.alpha, self.alpha) # beta distribution
        # l = 1 if self.epoch <= 120 else 0  # seperated stage

        mixed_feature = 2 * torch.cat((l * feature_a, (1 - l) * feature_b), dim=1)
        output = model(mixed_feature, classifier_flag=True)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )

        return loss, now_acc
