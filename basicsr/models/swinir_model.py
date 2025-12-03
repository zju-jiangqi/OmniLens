import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from collections import OrderedDict

@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def test(self):
        # pad to multiplication of window_size
        # window_size = self.opt['network_g']['window_size']
        # scale = self.opt.get('scale', 1)
        # mod_pad_h, mod_pad_w = 0, 0
        # _, _, h, w = self.lq.size()
        # if h % window_size != 0:
        #     mod_pad_h = window_size - h % window_size
        # if w % window_size != 0:
        #     mod_pad_w = window_size - w % window_size
        # img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                n = img.size(0)
                outs = []
                m = self.opt['val'].get('max_minibatch', n)
                i = 0
                while i < n:
                    j = i + m
                    if j >= n:
                        j = n
                    pred = self.net_g_ema(img[i:j, :, :, :])
                    # pred = preds[0]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    # print('pred .. size', pred.size())
                    outs.append(pred)
                    i = j
                self.output = torch.cat(outs, dim=0)
            # self.net_g_ema.train()
                # self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                n = img.size(0)
                outs = []
                m = self.opt['val'].get('max_minibatch', n)
                i = 0
                while i < n:
                    j = i + m
                    if j >= n:
                        j = n
                    pred = self.net_g(img[i:j, :, :, :])
                    # pred = preds[0]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    # print('pred .. size', pred.size())
                    outs.append(pred)
                    i = j
                self.output = torch.cat(outs, dim=0)
            self.net_g.train()

        # _, _, h, w = self.output.size()
        # self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq



