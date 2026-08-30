import os
import time
import warnings
import inspect
import hashlib
import json
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from core.data_processing.data_factory import data_provider
from core.exp.exp_basic import Exp_Basic
from core.utils.metrics import metric
from core.utils.tools import EarlyStopping, adaptive_adjust_lr

from models import DLinear, FADLinear, FITS, FilterTS, PatchTST, TimeMixer, WPMixer, itransformer

warnings.filterwarnings('ignore')



class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False

        if self.args.use_gpu:
            available_gpu_count = torch.cuda.device_count()
            requested_devices = self.args.devices if getattr(self.args, 'use_multi_gpu', False) else str(self.args.gpu)
            req_device_ids = list(map(int, requested_devices.split(',')))

            if max(req_device_ids) >= available_gpu_count:
                print(
                    f"Warning: Requested device_ids {req_device_ids} mismatch with available GPUs ({available_gpu_count}).")
                fixed_len = min(len(req_device_ids), available_gpu_count)
                self.args.device_ids = list(range(fixed_len))
                print(f"Autocorrected device_ids to: {self.args.device_ids}")
            else:
                self.args.device_ids = req_device_ids

            if len(self.args.device_ids) > 0:
                self.device = torch.device(f"cuda:{self.args.device_ids[0]}")
                self.args.gpu = self.args.device_ids[0]
            else:
                self.device = torch.device("cpu")
                self.args.use_gpu = False
        else:
            self.device = torch.device("cpu")

        self.args.device = self.device
        self.args.experiment_root = getattr(self.args, "experiment_root", "experiments_root")
        self.experiment_root = self.args.experiment_root
        self.exp_path = None

        self.model = self._build_model().to(self.device)
        self._analyze_model_signature()

    def _build_model(self):
        model_dict = {
            'FADLinear': FADLinear,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'TimeMixer': TimeMixer,
            'itransformer': itransformer,
            'WPMixer': WPMixer,
            'FilterTS': FilterTS,
            'FITS': FITS,
        }

        if self.args.model in model_dict:
            model = model_dict[self.args.model].Model(self.args).float()
        else:
            raise NotImplementedError(f"Model '{self.args.model}' not found in model_dict.")

        if self.args.use_gpu and getattr(self.args, 'use_multi_gpu', False) and len(self.args.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {self.args.model} | Params: {total_params / 1e6:.2f}M | Device: {self.device}")
        return model

    def _analyze_model_signature(self):
        if isinstance(self.model, nn.DataParallel):
            forward_func = self.model.module.forward
        else:
            forward_func = self.model.forward

        params = inspect.signature(forward_func).parameters

        
        self.need_x_mark = any(p in params for p in ['batch_x_mark', 'x_mark', 'x_mark_enc'])
        self.need_dec_inp = any(p in params for p in ['dec_inp', 'x_dec'])
        self.need_target = 'target' in params

        print(
            f"Model Signature Detected -> Needs x_mark: {self.need_x_mark}, "
            f"Needs dec_inp: {self.need_dec_inp}, Needs target: {self.need_target}"
        )

    def _feature_start_index(self):
        return -1 if self.args.features == 'MS' else 0

    def _first_model_output(self, outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _slice_target(self, batch_y):
        f_dim = self._feature_start_index()
        return batch_y[:, -self.args.pred_len:, f_dim:]

    def _slice_prediction(self, outputs):
        outputs = self._first_model_output(outputs)
        f_dim = self._feature_start_index()
        return outputs[:, -self.args.pred_len:, f_dim:]

    def _slice_outputs_target(self, outputs, batch_y, context):
        outputs = self._slice_prediction(outputs)
        target = self._slice_target(batch_y)
        if outputs.shape != target.shape:
            raise ValueError(
                f"{context}: output/target shape mismatch after features={self.args.features} slicing. "
                f"output={tuple(outputs.shape)}, target={tuple(target.shape)}. "
                f"For M tasks, set --c_out equal to --enc_in and use a model that returns all variables."
            )
        return outputs, target

    def _inverse_transform_array(self, scaler, data, context):
        if scaler is None:
            return data.copy()
        if data.ndim != 3:
            raise ValueError(f"{context}: expected 3D array [N, pred_len, C], got shape={data.shape}.")

        n_feat = int(scaler.n_features_in_)
        out_dim = int(data.shape[-1])
        if out_dim > n_feat:
            raise ValueError(
                f"{context}: output channel count {out_dim} exceeds scaler feature count {n_feat}."
            )

        flat = data.reshape(-1, out_dim)
        if out_dim == n_feat:
            restored = scaler.inverse_transform(flat)
        else:
            
            start = n_feat - out_dim
            dummy = np.zeros((flat.shape[0], n_feat), dtype=flat.dtype)
            dummy[:, start:] = flat
            restored = scaler.inverse_transform(dummy)[:, start:]

        return restored.reshape(data.shape)

    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(),
                                  lr=self.args.learning_rate,
                                  weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
        batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len], dec_inp], dim=1).to(self.device)

        return batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, target=None):
        kwargs = {}
        if self.need_target and target is not None:
            kwargs["target"] = target
        if self.need_x_mark and self.need_dec_inp:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, **kwargs)
        if self.need_x_mark:
            return self.model(batch_x, batch_x_mark, **kwargs)
        return self.model(batch_x, **kwargs)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                    self._process_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, target = self._slice_outputs_target(outputs, batch_y, 'vali')
                loss = criterion(outputs.detach(), target.detach()).item()

                total_loss.append(loss)
        self.model.train()
        return np.average(total_loss)

    def _train_standard(self, setting, checkpoint_subdir=None, fits_full_sequence=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_losses = [] if not self.args.train_only else None

        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        self.exp_path = os.path.join(self.experiment_root, setting)
        checkpoint_root = os.path.join(self.exp_path, 'checkpoints')
        path = os.path.join(checkpoint_root, checkpoint_subdir) if checkpoint_subdir else checkpoint_root
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        use_amp = self.args.use_amp
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]

                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                    self._process_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                target = self._slice_target(batch_y)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if fits_full_sequence:
                        outputs = self.model(batch_x, return_full_sequence=True)
                        target = torch.cat([batch_x, batch_y[:, -self.args.pred_len:, :]], dim=1)
                        loss = criterion(outputs, target)
                    else:
                        aux_target = batch_y[:, -self.args.pred_len:, :] if self.need_target else None
                        outputs = self._forward_model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            target=aux_target,
                        )
                        aux_loss = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
                        outputs, target = self._slice_outputs_target(outputs, batch_y, 'train')
                        loss = criterion(outputs, target)
                        if aux_loss is not None and torch.is_tensor(aux_loss):
                            loss = loss + aux_loss.mean()

                scaler.scale(loss).backward()
                scaler.unscale_(model_optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                scaler.step(model_optim)
                scaler.update()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if not self.args.train_only:
                current_vali_loss = self.vali(vali_data, vali_loader, criterion)
                if checkpoint_subdir is None:
                    test_loss = self.vali(test_data, test_loader, criterion)
                    test_message = f' Test Loss: {test_loss:.7f}'
                else:
                    test_message = ''
                if not np.isfinite(current_vali_loss):
                    raise FloatingPointError(f'Non-finite validation loss in {checkpoint_subdir or "standard"}: {current_vali_loss}')
                vali_losses.append(current_vali_loss)
                print(f'Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {current_vali_loss:.7f}{test_message}')
                early_stopping(current_vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adaptive_adjust_lr(model_optim, epoch + 1, self.args,
                               vali_losses if not self.args.train_only else None)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f'Best checkpoint not found: {best_model_path}')
        best_validation_loss = float(early_stopping.val_loss_min)
        if not np.isfinite(best_validation_loss):
            raise FloatingPointError(f'Non-finite best validation loss: {best_validation_loss}')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if checkpoint_subdir:
            final_best_path = os.path.join(path, 'best.pth')
            os.replace(best_model_path, final_best_path)
            checkpoint_hash = self._sha256_file(final_best_path)
            self._write_stage_manifest(path, setting, checkpoint_subdir, final_best_path, checkpoint_hash, best_validation_loss)
            return {'completed': True, 'best_checkpoint_path': final_best_path,
                    'best_validation_loss': best_validation_loss, 'best_checkpoint_sha256': checkpoint_hash}
        return self.model

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_stage_manifest(self, stage_dir, setting, stage, checkpoint_path, checkpoint_hash, val_loss):
        manifest = {'setting_id': setting, 'stage': stage, 'best_checkpoint': os.path.basename(checkpoint_path),
                    'best_checkpoint_sha256': checkpoint_hash, 'best_validation_loss': val_loss, 'completed': True}
        final_path = os.path.join(stage_dir, 'completed.json')
        temp_path = final_path + f'.tmp.{os.getpid()}.{uuid.uuid4().hex}'
        try:
            with open(temp_path, 'w', encoding='utf-8', newline='\n') as handle:
                json.dump(manifest, handle, ensure_ascii=False, separators=(',', ':'))
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_path, final_path)
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

    def _verify_stage_result(self, result, setting, stage):
        if not result.get('completed') or not np.isfinite(result.get('best_validation_loss', np.nan)):
            raise RuntimeError(f'Invalid FITS {stage} result: {result}')
        checkpoint_path = result['best_checkpoint_path']
        with open(os.path.join(os.path.dirname(checkpoint_path), 'completed.json'), 'r', encoding='utf-8') as handle:
            manifest = json.load(handle)
        actual_hash = self._sha256_file(checkpoint_path)
        if manifest.get('setting_id') != setting or manifest.get('stage') != stage:
            raise RuntimeError(f'FITS {stage} manifest identity mismatch.')
        if manifest.get('best_checkpoint_sha256') != actual_hash or result['best_checkpoint_sha256'] != actual_hash:
            raise RuntimeError(f'FITS {stage} checkpoint hash mismatch.')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return checkpoint_path

    def _promote_checkpoint(self, source_path, public_path):
        self.model.load_state_dict(torch.load(source_path, map_location=self.device))
        temp_path = public_path + f'.tmp.{os.getpid()}.{uuid.uuid4().hex}'
        committed = False
        try:
            torch.save(self.model.state_dict(), temp_path)
            verified = torch.load(temp_path, map_location='cpu')
            if set(verified.keys()) != set(self.model.state_dict().keys()):
                raise RuntimeError('Promoted checkpoint key mismatch.')
            os.replace(temp_path, public_path); committed = True
        finally:
            if not committed and os.path.exists(temp_path): os.remove(temp_path)

    def train(self, setting):
        if self.args.model != 'FITS' or self.args.fits_train_mode == 1:
            return self._train_standard(setting)
        if self.args.fits_train_mode != 2:
            raise ValueError('FITS train mode must be 1 or 2.')
        self.exp_path = os.path.join(self.experiment_root, setting)
        checkpoint_root = os.path.join(self.exp_path, 'checkpoints')
        if os.path.exists(checkpoint_root):
            raise FileExistsError(f'FITS mode 2 refuses existing checkpoint directory: {checkpoint_root}')
        stage1 = self._train_standard(setting, checkpoint_subdir='fits_stage1', fits_full_sequence=True)
        self._verify_stage_result(stage1, setting, 'fits_stage1')
        stage2 = self._train_standard(setting, checkpoint_subdir='fits_stage2', fits_full_sequence=False)
        stage2_path = self._verify_stage_result(stage2, setting, 'fits_stage2')
        self._promote_checkpoint(stage2_path, os.path.join(checkpoint_root, 'checkpoint.pth'))
        return self.model

    
    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        self.exp_path = os.path.join(self.experiment_root, setting)

        if test:
            print('loading model...')
            self.model.load_state_dict(torch.load(os.path.join(self.exp_path, 'checkpoints', 'checkpoint.pth'), map_location=self.device))

        preds, trues = [], []
        scaler = getattr(test_data, 'scaler', None)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]


                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                    self._process_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, batch_y = self._slice_outputs_target(outputs, batch_y, 'test')

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())


        preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

        
        preds_target = preds
        trues_target = trues
        preds_target_ori = self._inverse_transform_array(scaler, preds_target, 'test/preds')
        trues_target_ori = self._inverse_transform_array(scaler, trues_target, 'test/trues')

        
        def _ensure_scalar(x):
            if x is None: return 0.0
            if isinstance(x, np.ndarray):
                return float(np.nanmean(x)) if x.ndim > 0 else float(x.item())
            return float(x) if not hasattr(x, 'item') else float(x.item())

        raw_metrics_n = metric(preds_target, trues_target)
        mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n, corr_n = [_ensure_scalar(m) for m in raw_metrics_n]

        raw_metrics_o = metric(preds_target_ori, trues_target_ori)
        mae_o, mse_o, rmse_o, mape_o, mspe_o, rse_o, corr_o = [_ensure_scalar(m) for m in raw_metrics_o]

        metric_scope = 'all_variables' if self.args.features == 'M' else f'target={self.args.target}'
        print(f'【归一化尺度】mse:{mse_n:.4f}, mae:{mae_n:.4f}')
        print(f'【原始尺度】{metric_scope}, mse:{mse_o:.4f}, mae:{mae_o:.4f}')

        
        results_dir = os.path.join(self.exp_path, 'results')
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, 'metrics_norm_target.npy'),
                np.array([mae_n, mse_n, rmse_n, mape_n, mspe_n, rse_n, corr_n], dtype=np.float64))
        np.save(os.path.join(results_dir, 'metrics_ori_target.npy'),
                np.array([mae_o, mse_o, rmse_o, mape_o, mspe_o, rse_o, corr_o], dtype=np.float64))

        np.save(os.path.join(results_dir, 'pred_norm_target.npy'), preds_target)
        np.save(os.path.join(results_dir, 'true_norm_target.npy'), trues_target)
        np.save(os.path.join(results_dir, 'pred_ori_target.npy'), preds_target_ori)
        np.save(os.path.join(results_dir, 'true_ori_target.npy'), trues_target_ori)

        
        with open(os.path.join(self.experiment_root, 'global_result.txt'), 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"  [Normalized] MSE={mse_n:.6f}, MAE={mae_n:.6f}, RMSE={rmse_n:.6f}\n")
            f.write(f"  [Original]   MSE={mse_o:.6f}, MAE={mae_o:.6f}, RMSE={rmse_o:.6f}\n\n")

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        self.exp_path = os.path.join(self.experiment_root, setting)

        if load:
            checkpoint_path = os.path.join(self.exp_path, 'checkpoints', 'checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]

                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = \
                    self._process_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self._first_model_output(outputs)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        if getattr(pred_data, 'scale', False):
            preds = pred_data.inverse_transform(preds)

        results_dir = os.path.join(self.exp_path, 'results')
        os.makedirs(results_dir, exist_ok=True)

        np.save(os.path.join(results_dir, 'real_prediction.npy'), preds)

        if hasattr(pred_data, 'future_dates') and hasattr(pred_data, 'cols'):
            future_dates = np.expand_dims(pred_data.future_dates, axis=0)
            if len(future_dates.shape) == 1:
                future_dates = future_dates.reshape(-1, 1)
            try:
                report_df = pd.DataFrame(np.hstack([future_dates, preds[0]]), columns=pred_data.cols)
                report_df.to_csv(os.path.join(results_dir, 'real_prediction.csv'), index=False)
            except Exception as e:
                print(f"Warning: Could not save prediction CSV due to shape mismatch: {e}")

        return
