import random
import torch
from models.model import basemodel


class MTS2d_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)

        self.use_amp = (
            self.extra_params.get("enabled_amp", False)
            and torch.cuda.is_available()
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.replay_ratio = self.extra_params.get("replay_ratio", 0.5)

    def data_preprocess(self, data):
        if isinstance(data[0], list):
            data = data[0]

        x = data[0]   # [B, input_steps, C, H, W]
        y = data[1]   # [B, target_steps, C, H, W]

        B, T, C, H, W = x.shape

        inp = x.view(B, T * C, H, W).float().to(self.device)
        y = y.float().to(self.device)

        return inp, y

    def _try_sample_replay(self, rb, target_steps):
        try:
            sample_inp, sample_target, _ = rb.sample(
                batch_size=1,
                return_target=True,
                sample_num=target_steps
            )

            sample_inp = torch.from_numpy(sample_inp).float()

            if isinstance(sample_target, list):
                sample_target = sample_target[0]

            if not isinstance(sample_target, torch.Tensor):
                sample_target = torch.from_numpy(sample_target).float()

            if sample_target.dim() == 4:
                sample_target = sample_target.unsqueeze(0)

            return sample_inp, sample_target

        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"Replay sample skipped: {e}")
            return None

    def _try_store_replay(self, rb, inp, step):
        try:
            tar_idx = torch.tensor([[step]], dtype=torch.int32).numpy()
            rb.store(inp.detach().cpu().numpy(), tar_idx)
        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"Replay store skipped: {e}")

    def train_one_step(self, batch_data, step):
        rb = getattr(self, "replay_buff", None)

        # D’abord préparer le batch réel pour connaître target_steps
        inp, target_seq = self.data_preprocess(batch_data)

        target_steps = target_seq.shape[1]

        # Sampling replay si disponible
        if (
            rb is not None
            and hasattr(rb, "size")
            and rb.size > 0
            and random.random() < self.replay_ratio
        ):
            replay_batch = self._try_sample_replay(rb, target_steps)

            if replay_batch is not None:
                replay_inp, replay_target_seq = replay_batch
                inp = replay_inp.float().to(self.device)
                target_seq = replay_target_seq.float().to(self.device)

        model_key = list(self.model.keys())[0]
        optimizer = self.optimizer[model_key]

        optimizer.zero_grad()

        ar_steps = target_seq.shape[1]
        C = target_seq.shape[2]

        loss_total = 0.0
        current_input = inp

        for k in range(ar_steps):
            target = target_seq[:, k]

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    predict = self.model[model_key](current_input)

                    mean, log_var = torch.chunk(predict, 2, dim=1)
                    log_var = torch.clamp(log_var, -8, 5)

                    loss = (
                        torch.mean((mean - target) ** 2 * torch.exp(-log_var))
                        + torch.mean(log_var)
                    )
                    loss = loss / ar_steps

                self.scaler.scale(loss).backward()

            else:
                predict = self.model[model_key](current_input)

                mean, log_var = torch.chunk(predict, 2, dim=1)
                log_var = torch.clamp(log_var, -8, 5)

                loss = (
                    torch.mean((mean - target) ** 2 * torch.exp(-log_var))
                    + torch.mean(log_var)
                )
                loss = loss / ar_steps

                loss.backward()

            loss_total += loss.item()

            current_input = torch.cat(
                [current_input[:, C:].detach(), mean.detach()],
                dim=1
            )

        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

        # Stocker le batch réel dans replay buffer
        if rb is not None:
            self._try_store_replay(rb, inp, step)

        return {self.loss_type: loss_total}

    def test_one_step(self, batch_data):
        inp, target_seq = self.data_preprocess(batch_data)

        model_key = list(self.model.keys())[0]

        ar_steps = target_seq.shape[1]
        C = target_seq.shape[2]

        loss_total = 0.0
        current_input = inp

        for k in range(ar_steps):
            target = target_seq[:, k]

            predict = self.model[model_key](current_input)

            mean, log_var = torch.chunk(predict, 2, dim=1)
            log_var = torch.clamp(log_var, -8, 5)

            loss = (
                torch.mean((mean - target) ** 2 * torch.exp(-log_var))
                + torch.mean(log_var)
            )

            loss_total += loss.item()

            current_input = torch.cat(
                [current_input[:, C:], mean],
                dim=1
            )

        return {self.loss_type: loss_total / ar_steps}