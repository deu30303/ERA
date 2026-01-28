from __future__ import annotations

import inspect
from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl import DPOTrainer


# -------------------------
# Evidential head (quadrant logits)
# -------------------------
class EvidentialQuadrantHead(nn.Module):
    """
    hidden -> logits(4)
    logits order default: [kg, kn, ug, un]
    """
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(hidden))  # [B,4] logits


# -------------------------
# Stable EDL digamma loss (float32, clamp, device-safe annealing)
# -------------------------
def _edl_kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = alpha.device
    dtype = alpha.dtype
    ones = torch.ones((1, num_classes), device=device, dtype=dtype)
    sum_alpha = alpha.sum(dim=1, keepdim=True)

    first = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=1, keepdim=True)
    return first + second


def edl_digamma_loss_stable(
    logits: torch.Tensor,
    target_onehot: torch.Tensor,
    epoch_num: int,
    num_classes: int,
    annealing_step: int,
    evidence_fn=F.softplus,
    alpha_max: float = 10.0,
) -> torch.Tensor:

    logits = torch.clamp(logits, min=-10.0, max=10.0)

    logits32 = logits.float()
    y32 = target_onehot.float()

    logits32 = torch.clamp(logits32, min=-10.0, max=10.0)


    evidence = evidence_fn(logits32)
    
    alpha = evidence + 1.0
    

    alpha = torch.clamp(alpha, min=1.0001, max=alpha_max) 

    S = alpha.sum(dim=1, keepdim=True)
    S = torch.clamp(S, max=alpha_max * num_classes) # 합계도 제한

    term_fit = (torch.digamma(S) - torch.digamma(alpha))
    
    term_fit = torch.nan_to_num(term_fit, nan=0.0)
    
    A = (y32 * term_fit).sum(dim=1, keepdim=True)


    coef = min(1.0, float(epoch_num) / float(max(1, annealing_step)))
    annealing_coef = torch.tensor(coef, device=alpha.device, dtype=alpha.dtype)

    kl_alpha = (alpha - 1.0) * (1.0 - y32) + 1.0
    

    kl_alpha = torch.clamp(kl_alpha, min=1.0001, max=alpha_max)
    
    kl = _edl_kl_divergence(kl_alpha, num_classes)
    
    kl = torch.nan_to_num(kl, nan=0.0, posinf=100.0)

    return (A + annealing_coef * kl).mean()

class Trainer(DPOTrainer):

    def __init__(
        self,
        # ----- DS weighting -----
        use_ds: bool = True,
        gamma_ds: float = 1.0,
        kappa_clip: float = 0.999,
        eps_ds: float = 1e-8,
        default_discount: bool = True,  
        denom_floor: float = 1e-4,    
        max_grad_norm=1.0,

        # ----- Evidential heads -----
        use_evidential_quadrant_heads: bool = True,
        evidential_dropout: float = 0.0,
        share_quadrant_head: bool = False,
        idx_kg: int = 0,
        idx_kn: int = 1,
        idx_ug: int = 2,
        idx_un: int = 3,


        derive_param_prompt_from_prompt: bool = True,

        # ----- EDL loss -----
        use_edl_loss: bool = True,
        lambda_edl_rag: float = 0.1,
        lambda_edl_param: float = 0.1,
        annealing_step: int = 2000,
        alpha_max: float = 10,
        evidence_fn: str = "softplus",  # "softplus" | "relu"

        # ----- Optional SFT aux -----
        use_sft_aux: bool = True,
        coe_sft: float = 1.0,

        *args,
        **kwargs,
    ):
        if "ref_model" in kwargs and kwargs["ref_model"] is False:
            kwargs["ref_model"] = None

        sig = inspect.signature(DPOTrainer.__init__)
        allowed = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = sorted(set(kwargs.keys()) - set(filtered_kwargs.keys()))
        if dropped:
            print("[Trainer] Dropped unsupported kwargs:", dropped)

        super().__init__(*args, **filtered_kwargs)

        # DS config
        self.use_ds = use_ds
        self.gamma_ds = gamma_ds
        self.kappa_clip = kappa_clip
        self.eps_ds = eps_ds
        self.default_discount = default_discount
        self.denom_floor = denom_floor

        # indices
        self.idx_kg, self.idx_kn, self.idx_ug, self.idx_un = idx_kg, idx_kn, idx_ug, idx_un

        # Option B
        self.derive_param_prompt_from_prompt = derive_param_prompt_from_prompt

        # EDL config
        self.use_edl_loss = use_edl_loss
        self.lambda_edl_rag = lambda_edl_rag
        self.lambda_edl_param = lambda_edl_param
        self.annealing_step = annealing_step
        self.alpha_max = alpha_max
        self.evidence_fn = F.softplus if evidence_fn == "softplus" else F.relu

        # Optional SFT aux
        self.use_sft_aux = use_sft_aux
        self.coe_sft = coe_sft

        # Heads
        self.use_evidential_quadrant_heads = use_evidential_quadrant_heads
        self.share_quadrant_head = share_quadrant_head

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "word_embed_proj_dim"):
            hidden_size = self.model.config.word_embed_proj_dim
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model.config.")

        self.q_head_rag = EvidentialQuadrantHead(hidden_size, dropout=evidential_dropout).to(self.accelerator.device)
        self.q_head_param = self.q_head_rag if share_quadrant_head else EvidentialQuadrantHead(hidden_size, dropout=evidential_dropout).to(self.accelerator.device)

    # -------------------------
    # safe masked mean for logging
    # -------------------------
    @staticmethod
    def _safe_masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # returns scalar
        if mask.sum() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        return x[mask].mean()

    # -------------------------
    # concatenated chosen+rejected forward (one pass)
    # -------------------------
    def concatenated_forward(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_ref_model: bool = False,
        **kwargs,
        ): 
            """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

            We do this to avoid doing two forward passes, because it's faster for FSDP.
            """
            num_examples = batch["prompt_input_ids"].shape[0]

            concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

            model_kwargs = {}
            if self.aux_loss_enabled:
                model_kwargs["output_router_logits"] = True

            # Add the pixel values and attention masks for vision models
            if "pixel_values" in concatenated_batch:
                model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

            prompt_input_ids = concatenated_batch["prompt_input_ids"]
            prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
            completion_input_ids = concatenated_batch["completion_input_ids"]
            completion_attention_mask = concatenated_batch["completion_attention_mask"]
            if self.is_encoder_decoder:
                labels = completion_input_ids
                labels[completion_attention_mask == 0] = self.label_pad_token_id
                outputs = model(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    labels=labels,  # we need the labels for the logits to be returned
                    **model_kwargs,
                )
                logits = outputs.logits
                loss_mask = completion_attention_mask.bool()
            else:
                # Concatenate the prompt and completion inputs
                input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
                attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
                # Mask the prompt but not the completion for the loss
                loss_mask = torch.cat(
                    (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                    dim=1,
                )

                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                for i in range(attention_mask.size(0)):
                    first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                    input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                    attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                    loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

                # Get the first column idx that is all zeros and remove every column after that
                empty_cols = torch.sum(attention_mask, dim=0) == 0
                first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
                input_ids = input_ids[:, : first_empty_col - 1]
                attention_mask = attention_mask[:, : first_empty_col - 1]
                loss_mask = loss_mask[:, : first_empty_col - 1]

                # Truncate right
                if self.args.max_length is not None:
                    input_ids = input_ids[:, : self.args.max_length]
                    attention_mask = attention_mask[:, : self.args.max_length]
                    loss_mask = loss_mask[:, : self.args.max_length]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

                # Offset the logits by one to align with the labels
                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:].clone()
                loss_mask = loss_mask[:, 1:].bool()

            if logits.shape[:2] != labels.shape[:2]:
                # for llava, the returned logits include the image tokens (placed before the text tokens)
                seq_len = labels.shape[1]
                logits = logits[:, -seq_len:]

            # Compute the log probabilities of the labels
            labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            per_token_logps[~loss_mask] = 0
            all_logps = per_token_logps.sum(-1)

            output = {}

            output["chosen_logps"] = all_logps[:num_examples]
            output["rejected_logps"] = all_logps[num_examples:]
            output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
            output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()
            output["sft_logps"] = all_logps[:num_examples] / loss_mask[:num_examples].sum(-1)

            return output

    # -------------------------
    # last valid token hidden
    # -------------------------
    def _last_token_hidden(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs = out.hidden_states[-1]  # [B,T,H]
        last_idx = attention_mask.sum(dim=1) - 1
        last_idx = torch.clamp(last_idx, min=0)  # 안전장치
        B = hs.size(0)
        return hs[torch.arange(B, device=hs.device), last_idx]

    # -------------------------
    # Option B: derive param prompt from Llama3-base prompt
    # -------------------------
    def _find_subseq(self, haystack: List[int], needle: List[int]) -> int:
        n = len(needle)
        if n == 0 or n > len(haystack):
            return -1
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                return i
        return -1

    def _derive_param_prompt_llama3_base(
        self,
        prompt_input_ids: torch.Tensor,       # [B,T] left-padded
        prompt_attention_mask: torch.Tensor,  # [B,T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.processing_class
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        ctx_marker = tok.encode("### Contexts\n", add_special_tokens=False)
        q_marker = tok.encode("### Question\n", add_special_tokens=False)

        B, T = prompt_input_ids.shape
        out_ids = torch.full((B, T), pad_id, device=prompt_input_ids.device, dtype=prompt_input_ids.dtype)
        out_attn = torch.zeros((B, T), device=prompt_attention_mask.device, dtype=prompt_attention_mask.dtype)

        for b in range(B):
            ids = prompt_input_ids[b][prompt_attention_mask[b].bool()].tolist()

            i_ctx = self._find_subseq(ids, ctx_marker)
            i_q = self._find_subseq(ids, q_marker)

            if i_ctx != -1 and i_q != -1 and i_ctx < i_q:
                new_ids = ids[:i_ctx] + ids[i_q:]  # contexts 제거
            else:
                new_ids = ids

            new_len = min(len(new_ids), T)
            out_ids[b, T - new_len : T] = torch.tensor(new_ids[-new_len:], device=out_ids.device, dtype=out_ids.dtype)
            out_attn[b, T - new_len : T] = 1

        return out_ids, out_attn

    # -------------------------
    # alpha4 -> alpha2 (A=kg+ug, ¬A=kn+un)
    # -------------------------
    def _alpha4_to_alpha2(self, alpha4: torch.Tensor) -> torch.Tensor:
        e4 = torch.clamp(alpha4 - 1.0, min=0.0)
        eA = e4[:, self.idx_kg] + e4[:, self.idx_ug]
        eN = e4[:, self.idx_kn] + e4[:, self.idx_un]
        return torch.stack([eA + 1.0, eN + 1.0], dim=-1)

    # -------------------------
    # alpha2 -> DS mass
    # -------------------------
    def _alpha2_to_mass(self, alpha2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = torch.clamp(alpha2 - 1.0, min=0.0)
        eA, eN = e[:, 0], e[:, 1]
        E = eA + eN
        denom = E + 2.0
        mA = eA / denom
        mN = eN / denom
        mO = 2.0 / denom
        return mA, mN, mO

    def _discount(self, mA, mN, mO, w: torch.Tensor):
        w = torch.clamp(w, 0.0, 1.0)
        return w * mA, w * mN, (1.0 - w) + w * mO

    def _kappa(self, mpA, mpN, mrA, mrN):
        k = mpA * mrN + mpN * mrA
        return torch.clamp(k, 0.0, self.kappa_clip)

    def _dempster_combine(self, mpA, mpN, mpO, mrA, mrN, mrO, kappa):
        den = torch.clamp(1.0 - kappa, min=self.denom_floor) + self.eps_ds
        
        mA_num = (mpA * mrA + mpA * mrO + mpO * mrA)
        mN_num = (mpN * mrN + mpN * mrO + mpO * mrN)
        mO_num = (mpO * mrO)
        
        mA_num = torch.nan_to_num(mA_num, nan=0.0)
        mN_num = torch.nan_to_num(mN_num, nan=0.0)
        mO_num = torch.nan_to_num(mO_num, nan=0.0)

        mA = mA_num / den
        mN = mN_num / den
        mO = mO_num / den
        

        mA = torch.nan_to_num(mA, nan=0.0)
        mN = torch.nan_to_num(mN, nan=0.0)
        mO = torch.nan_to_num(mO, nan=1.0) 
        
        return mA, mN, mO

    # -------------------------
    # main loss + metrics
    # -------------------------
    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics: Dict[str, Any] = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # 1) DPO forward
        model_output = self.concatenated_forward(model, batch)

        # 2) reference logps
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)


        def clamp_logps(tensor):
            tensor = torch.nan_to_num(tensor, nan=-100.0, neginf=-100.0, posinf=0.0)
            return torch.clamp(tensor, min=-100.0, max=100.0)

        model_output["chosen_logps"] = clamp_logps(model_output["chosen_logps"])
        model_output["rejected_logps"] = clamp_logps(model_output["rejected_logps"])
        ref_chosen_logps = clamp_logps(ref_chosen_logps)
        ref_rejected_logps = clamp_logps(ref_rejected_logps)
        # ---------------------------------------------------------------------

        # 3) DPO loss per-sample
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
        )

        # 4) Option B: derive param prompt on the fly (if missing)
        if self.derive_param_prompt_from_prompt and ("param_prompt_input_ids" not in batch):
            if ("prompt_input_ids" in batch) and ("prompt_attention_mask" in batch):
                pids, pattn = self._derive_param_prompt_llama3_base(
                    batch["prompt_input_ids"],
                    batch["prompt_attention_mask"],
                )
                batch["param_prompt_input_ids"] = pids
                batch["param_prompt_attention_mask"] = pattn

        # 5) Build logits4 + alpha4 for rag/param
        logits4_rag = None
        logits4_param = None
        alpha4_rag = None
        alpha4_param = None

        if self.use_evidential_quadrant_heads:
            if ("prompt_input_ids" in batch) and ("prompt_attention_mask" in batch):
                h_rag = self._last_token_hidden(model, batch["prompt_input_ids"], batch["prompt_attention_mask"])
                logits4_rag = self.q_head_rag(h_rag)  # [B,4]
                # Logit Clamping is done inside edl_digamma_loss_stable, 
                # but we also clamp here for DS calculation to be safe
                logits4_rag = torch.clamp(logits4_rag, -15.0, 15.0) 
                alpha4_rag = self.evidence_fn(logits4_rag.float()) + 1.0
                alpha4_rag = torch.clamp(alpha4_rag, 1.0, self.alpha_max)

            if ("param_prompt_input_ids" in batch) and ("param_prompt_attention_mask" in batch):
                h_par = self._last_token_hidden(model, batch["param_prompt_input_ids"], batch["param_prompt_attention_mask"])
                logits4_param = self.q_head_param(h_par)  # [B,4]
                logits4_param = torch.clamp(logits4_param, -15.0, 15.0)
                alpha4_param = self.evidence_fn(logits4_param.float()) + 1.0
                alpha4_param = torch.clamp(alpha4_param, 1.0, self.alpha_max)

        # 6) DS weighting (needs both sources)
        if self.use_ds and (alpha4_rag is not None) and (alpha4_param is not None):
            # DS는 float32로 (안정)
            alpha2_r = self._alpha4_to_alpha2(alpha4_rag)    # [B,2] float32
            alpha2_p = self._alpha4_to_alpha2(alpha4_param)  # [B,2] float32

            mrA, mrN, mrO = self._alpha2_to_mass(alpha2_r)
            mpA, mpN, mpO = self._alpha2_to_mass(alpha2_p)

            # reliabilities wp/wr
            if "wp" in batch:
                wp = batch["wp"].to(mrO.device).float()
                if wp.dim() == 0:
                    wp = wp.expand_as(mpO)
            elif self.default_discount:
                wp = torch.clamp(1.0 - mpO, 0.0, 1.0)
            else:
                wp = torch.ones_like(mpO)

            if "wr" in batch:
                wr = batch["wr"].to(mrO.device).float()
                if wr.dim() == 0:
                    wr = wr.expand_as(mrO)
            elif self.default_discount:
                wr = torch.clamp(1.0 - mrO, 0.0, 1.0)
            else:
                wr = torch.ones_like(mrO)

            # discount
            mpA, mpN, mpO = self._discount(mpA, mpN, mpO, wp)
            mrA, mrN, mrO = self._discount(mrA, mrN, mrO, wr)
            kappa = self._kappa(mpA, mpN, mrA, mrN)
            
            kappa = torch.nan_to_num(kappa, nan=0.0, posinf=1.0, neginf=0.0)
            mrO = torch.nan_to_num(mrO, nan=1.0)

            mA_op, mN_op, mO_op = self._dempster_combine(mpA, mpN, mpO, mrA, mrN, mrO, kappa)

            # weight = 1 + γ κ (1 - m_r'(Ω))
            ds_weight = 1.0 + float(self.gamma_ds) * kappa * (1.0 - mrO)
            
            ds_weight = torch.nan_to_num(ds_weight, nan=1.0, posinf=1.0, neginf=1.0)

            losses = losses * ds_weight.to(losses.dtype)

            # metrics
            metrics[f"{prefix}ds/kappa_mean"] = kappa.detach().mean().cpu()
            metrics[f"{prefix}ds/weight_mean"] = ds_weight.detach().mean().cpu()
            metrics[f"{prefix}ds/mrO_mean"] = mrO.detach().mean().cpu()
            metrics[f"{prefix}ds/mO_op_mean"] = mO_op.detach().mean().cpu()
            metrics[f"{prefix}ds/mA_op_mean"] = mA_op.detach().mean().cpu()

            # evidence sums for monitoring
            e4p = torch.clamp(alpha4_param - 1.0, min=0.0)
            e4r = torch.clamp(alpha4_rag - 1.0, min=0.0)
            metrics[f"{prefix}ds/e4_param_sum_mean"] = e4p.sum(dim=-1).detach().mean().cpu()
            metrics[f"{prefix}ds/e4_rag_sum_mean"] = e4r.sum(dim=-1).detach().mean().cpu()

        # 7) EDL digamma loss (rag/param) using quadrant label
        if self.use_edl_loss and ("label" in batch):
            labels = batch["label"].to(losses.device)
            y = F.one_hot(labels, num_classes=4).float()

            epoch_num = int(getattr(self.state, "epoch", 0) or 0)

            edl_total = torch.zeros((), device=losses.device, dtype=torch.float32)

            if logits4_rag is not None and self.lambda_edl_rag > 0:
                edl_rag = edl_digamma_loss_stable(
                    logits=logits4_rag,
                    target_onehot=y,
                    epoch_num=epoch_num,
                    num_classes=4,
                    annealing_step=self.annealing_step,
                    evidence_fn=self.evidence_fn,
                    alpha_max=self.alpha_max,
                )
                edl_total = edl_total + float(self.lambda_edl_rag) * edl_rag
                metrics[f"{prefix}edl/rag"] = edl_rag.detach().cpu()

            if logits4_param is not None and self.lambda_edl_param > 0:
                edl_param = edl_digamma_loss_stable(
                    logits=logits4_param,
                    target_onehot=y,
                    epoch_num=epoch_num,
                    num_classes=4,
                    annealing_step=self.annealing_step,
                    evidence_fn=self.evidence_fn,
                    alpha_max=self.alpha_max,
                )
                edl_total = edl_total + float(self.lambda_edl_param) * edl_param
                metrics[f"{prefix}edl/param"] = edl_param.detach().cpu()


            edl_total = torch.nan_to_num(edl_total, nan=0.0)


            losses = losses + edl_total.to(losses.dtype)
            metrics[f"{prefix}edl/total"] = edl_total.detach().cpu()

        # 8) Optional SFT aux 
        if self.use_sft_aux and self.coe_sft > 0:
            sft_logps = model_output["sft_logps"]
            

            sft_logps = torch.clamp(sft_logps, min=-10.0)
            sft_logps = torch.nan_to_num(sft_logps, nan=-10.0, neginf=-10.0, posinf=0.0)
            
            sft_loss_vec = -sft_logps 
            
            losses = losses + float(self.coe_sft) * sft_loss_vec
            
            metrics[f"{prefix}sft_loss"] = sft_loss_vec.mean().detach().cpu()

        # 9) Base DPO metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().detach().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().detach().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().detach().cpu()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()

        return losses.mean(), metrics