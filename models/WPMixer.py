import torch.nn as nn
import torch
from core.modules.dwt_decomposition import Decomposition


class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class Mixer(nn.Module):
    def __init__(self, input_seq=[], out_seq=[], batch_size=[], channel=[], d_model=[],
                 dropout=[], tfactor=[], dfactor=[]):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.tMixer = TokenMixer(input_seq=self.input_seq, batch_size=self.batch_size, channel=self.channel,
                                 pred_seq=self.pred_seq, dropout=self.dropout, factor=self.tfactor,
                                 d_model=self.d_model)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)
        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    def __init__(self, input_seq=[], pred_seq=[], batch_size=[], channel=[], d_model=[],
                 dropout=[], embedding_dropout=[], tfactor=[], dfactor=[], patch_len=[], patch_stride=[]):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)
        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)
        self.mixer1 = Mixer(input_seq=self.patch_num, out_seq=self.patch_num, batch_size=self.batch_size,
                            channel=self.channel, d_model=self.d_model, dropout=self.dropout,
                            tfactor=self.tfactor, dfactor=self.dfactor)
        self.mixer2 = Mixer(input_seq=self.patch_num, out_seq=self.patch_num, batch_size=self.batch_size,
                            channel=self.channel, d_model=self.d_model, dropout=self.dropout,
                            tfactor=self.tfactor, dfactor=self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))

    def forward(self, x):
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))
        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)
        out = self.head(out)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch


class WPMixerCore(nn.Module):
    def __init__(self, input_length=[], pred_length=[], wavelet_name=[], level=[], batch_size=[], channel=[],
                 d_model=[], dropout=[], embedding_dropout=[], tfactor=[], dfactor=[], device=[],
                 patch_len=[], patch_stride=[], no_decomposition=[], use_amp=[]):
        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        self.Decomposition_model = Decomposition(input_length=self.input_length, pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name, level=self.level,
                                                 batch_size=self.batch_size, channel=self.channel,
                                                 d_model=self.d_model, tfactor=self.tfactor, dfactor=self.dfactor,
                                                 device=self.device, no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp)
        self.input_w_dim = self.Decomposition_model.input_w_dim
        self.pred_w_dim = self.Decomposition_model.pred_w_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq=self.input_w_dim[i],
                                            pred_seq=self.pred_w_dim[i], batch_size=self.batch_size,
                                            channel=self.channel, d_model=self.d_model, dropout=self.dropout,
                                            embedding_dropout=self.embedding_dropout, tfactor=self.tfactor,
                                            dfactor=self.dfactor, patch_len=self.patch_len,
                                            patch_stride=self.patch_stride)
                                            for i in range(len(self.input_w_dim))])

    def forward(self, xL):
        x = xL.transpose(1, 2)
        xA, xD = self.Decomposition_model.transform(x)
        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)
        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]
        return xT


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.task_name = args.task_name
        tfactor = getattr(args, 'tfactor', 5)
        dfactor = getattr(args, 'dfactor', 5)
        wavelet = getattr(args, 'wavelet', 'db2')
        level = getattr(args, 'level', 1)
        stride = getattr(args, 'stride', 8)
        embedding_dropout = getattr(args, 'embedding_dropout', None)
        if embedding_dropout is None:
            embedding_dropout = args.dropout
        no_decomposition = getattr(args, 'no_decomposition', False)

        self.wpmixerCore = WPMixerCore(input_length=args.seq_len, pred_length=args.pred_len,
                                       wavelet_name=wavelet, level=level, batch_size=args.batch_size,
                                       channel=args.enc_in, d_model=args.d_model, dropout=args.dropout,
                                       embedding_dropout=embedding_dropout, tfactor=tfactor, dfactor=dfactor,
                                       device=args.device, patch_len=args.patch_len, patch_stride=stride,
                                       no_decomposition=no_decomposition, use_amp=args.use_amp)

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.args.c_out:]
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
