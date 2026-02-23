import jax
import jax.numpy as jnp

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils, nets
from . import ninjax as nj
import numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd
import jax, jax.numpy as jnp
from . import ninjax as nj, jaxutils, nets

class Disag(nj.Module):
    def __init__(self, wm, act_space, config):
        self.config = config.update({"disag_head.inputs": ["tensor"]})
        self.opt = jaxutils.Optimizer(name="disag_opt", **config.expl_opt)
        self.inputs = nets.Input(config.disag_head.inputs, dims="deter")
        self.target = nets.Input(self.config.disag_target, dims="deter")
        self.nets = [nets.MLP(shape=None, **self.config.disag_head, name=f"disag{i}") for i in range(self.config.disag_models)]

    def __call__(self, traj):
        inp = self.inputs(traj)
        preds = jnp.array([net(inp).mode() for net in self.nets])
        return preds.std(0).mean(-1)[1:]

    def train(self, data):
        ctx = nj.context()
        if not hasattr(self, "_init_done"):
            prev_flag = ctx.create
            ctx.create  = True
            _ = self.loss(data)          
            ctx.create  = prev_flag
            self._init_done = True
        return self.opt(self.nets, self.loss, data)

    def loss(self, data):
        inp = sg(self.inputs(data)[:, :-1])
        tar = sg(self.target(data)[:, 1:])
        losses = []
        for net in self.nets:
            net._shape = tar.shape[2:]
            losses.append(-net(inp).log_prob(tar).mean())
        return jnp.array(losses).sum()
    

class RND(nj.Module):
    """Random‑Network Distillation intrinsic reward.

    target: fixed random MLP (no grads)
    predictor: trainable MLP that learns to match target
    reward_t = || predictor(embed_t) - target(embed_t) ||_2
    """

    def __init__(self, wm, act_space, config, name="rnd_reward"):
        self.config = config
        feat_dim = config.rnd_feat_dim          
        self.target = nets.MLP(shape=None, **self.config.rnd_head, name="target")
        self.pred = nets.MLP(shape=None, **self.config.rnd_head, name="predictor")
        self.opt = jaxutils.Optimizer(name="rnd_opt", **config.rnd_opt)
        self.mom = jaxutils.Moments(**config.rnd_norm, name="rnd_norm")

    def __call__(self, traj):
        feats = traj["embed"] if "embed" in traj else traj["deter"]
        # feats = traj["deter"]
        feats = feats[:-1] 
        tgt   = self.target(feats)
        pred  = self.pred(feats)
        rew   = jnp.mean(jnp.square(pred - tgt), axis=-1)  
        mean, invstd = self.mom(rew)
        rew_norm = (rew - mean) * invstd
        return rew_norm

    def train(self, data):
        print(data.keys())
        feats = jax.lax.stop_gradient(data["embed"])     
        tgt   = self.target(feats)     
        def loss_fn():                                  
            pred = self.pred(feats)     
            loss = jnp.square(pred - tgt).mean()               
            return loss.astype(jnp.float32) 
        mets = self.opt(self.pred, loss_fn)  
        loss_rnd = mets[f"{self.opt.name}_loss"]       
        return {"loss_rnd": loss_rnd}
    
class ICM(nj.Module):
    def __init__(self, wm, act_space, config, name="icm"):
        self.act_dim = int(np.prod(act_space.shape))
        if not act_space.discrete:
            raise NotImplementedError("ICM inverse model assumes discrete one-hot actions.")
        self.beta = config.icm_beta                    # blend in loss
        self.eta  = getattr(config, "icm_eta", 1.0)    # scale for reward only

        inv_kw = dict(config.icm_head)
        inv_kw["dist"] = "onehot"
        self.inv = nets.MLP(shape=(self.act_dim,), **inv_kw, name="inv")

        self.fwd = nets.MLP(shape=None, **config.icm_head, name="fwd")
        self.opt = jaxutils.Optimizer(name="icm_opt", **config.icm_opt)
        self.mom = jaxutils.Moments(**config.icm_norm, name="icm_norm")

    def __call__(self, traj):
        def _flat(x): return x.reshape(x.shape[:2] + (-1,))
        z  = jnp.concatenate([traj["deter"], _flat(traj["stoch"])], -1)  # [T+1,B,Dz]
        z_t, z_tp1 = z[:-1], z[1:]
        a_t = traj["action"][:-1]                                        # [T,B,A]

        # forward prediction
        if getattr(self.fwd, "_shape", None) != z_tp1.shape[2:]:
            self.fwd._shape = z_tp1.shape[2:]
        pred = self.fwd(jnp.concatenate([z_t, a_t], -1)).mean()          # [T,B,Dz]

        fwd_err = 0.5 * jnp.square(pred - z_tp1).mean(-1)                # [T,B]
        mean, invstd = self.mom(fwd_err)
        return self.eta * (fwd_err - mean) * invstd                      # intrinsic reward
    
    def train(self, data):
        def _flat(x): return x.reshape(x.shape[:2] + (-1,))
        # B,T,...  -> latents and actions
        z = jnp.concatenate([data["deter"], _flat(data["stoch"])], -1)  # [B, T+1, Dz]
        a = data["action"]                                              # [B, T,   A]
        z, a = jax.lax.stop_gradient(z), jax.lax.stop_gradient(a)

        # -> time-major
        z = jnp.swapaxes(z, 0, 1)  # [Tz+1, B, Dz]
        a = jnp.swapaxes(a, 0, 1)  # [Ta,   B, A]

        # align sequence lengths
        T = min(z.shape[0] - 1, a.shape[0])    # python ints, safe for slicing
        if T == 0:
            return {"icm_inv_loss": jnp.array(0.0, jnp.float32),
                    "icm_fwd_loss": jnp.array(0.0, jnp.float32)}

        z_t   = z[:T]          # [T, B, Dz]
        z_tp1 = z[1:T+1]       # [T, B, Dz]
        a_t   = a[:T]          # [T, B, A]

        # mask across episode boundaries
        if "cont" in data:
            cont = jnp.swapaxes(data["cont"], 0, 1)[:T]   # [T, B]
        else:
            cont = jnp.ones((T, z.shape[1]), jnp.float32) # [T, B]

        def loss_fn():
            # inverse loss
            inv_in   = jnp.concatenate([z_t, z_tp1], -1)          # [T,B,2*Dz]
            inv_dist = self.inv(inv_in)                           # OneHot
            inv_loss_step = -inv_dist.log_prob(a_t)               # [T,B]

            # forward loss
            if getattr(self.fwd, "_shape", None) != z_tp1.shape[2:]:
                self.fwd._shape = z_tp1.shape[2:]
            fwd_in   = jnp.concatenate([z_t, a_t], -1)            # [T,B,Dz+A]
            pred     = self.fwd(fwd_in).mean()                    # [T,B,Dz]
            fwd_loss_step = jnp.square(pred - z_tp1).mean(-1)     # [T,B]

            denom   = jnp.maximum(cont.sum(), 1.0)
            inv_loss = (inv_loss_step * cont).sum() / denom
            fwd_loss = (fwd_loss_step * cont).sum() / denom

            total = (1.0 - self.beta) * inv_loss + self.beta * fwd_loss
            return total.astype(jnp.float32), {
                "icm_inv_loss": inv_loss,
                "icm_fwd_loss": fwd_loss,
            }

        opt_mets, aux = self.opt([self.inv, self.fwd], loss_fn, has_aux=True)
        return {**opt_mets, **aux}


    # def train(self, data):
    #     def _flat(x): return x.reshape(x.shape[:2] + (-1,))
    #     # [B,T,...] -> latents and actions, then stop-grad
    #     z  = jnp.concatenate([data["deter"], _flat(data["stoch"])], -1)  # [B,T+1,Dz]
    #     a  = data["action"]                                              # [B,T,A]
    #     z, a = jax.lax.stop_gradient(z), jax.lax.stop_gradient(a)

    #     # # to time-major
    #     # z, a = jnp.swapaxes(z, 0, 1), jnp.swapaxes(a, 0, 1)              # [T+1,B,Dz], [T,B,A]
    #     # z_t, z_tp1, a_t = z[:-1], z[1:], a

    #     # align lengths for (z_t, a_t, z_{t+1})
    #     Lz, La = z.shape[0], a.shape[0]
    #     T = min(Lz - 1, La)            # transitions count
    #     z_t   = z[:T]                   # [T,B,Dz]
    #     z_tp1 = z[1:T+1]                # [T,B,Dz]
    #     a_t   = a[:T]                   # [T,B,A]

    #     # mask across episode boundaries if available
    #     if "cont" in data:
    #         cont = jnp.swapaxes(data["cont"][:, 1:], 0, 1)               # [T,B]
    #     else:
    #         cont = jnp.ones(a_t.shape[:2], jnp.float32)

    #     def loss_fn():
    #         # inverse loss
    #         inv_dist = self.inv(jnp.concatenate([z_t, z_tp1], -1))
    #         inv_loss_step = -inv_dist.log_prob(a_t)                       # [T,B]

    #         # forward loss
    #         if getattr(self.fwd, "_shape", None) != z_tp1.shape[2:]:
    #             self.fwd._shape = z_tp1.shape[2:]
    #         pred = self.fwd(jnp.concatenate([z_t, a_t], -1)).mean()       # [T,B,Dz]
    #         fwd_loss_step = jnp.square(pred - z_tp1).mean(-1)             # [T,B]

    #         denom = jnp.maximum(cont.sum(), 1.0)
    #         inv_loss = (inv_loss_step * cont).sum() / denom
    #         fwd_loss = (fwd_loss_step * cont).sum() / denom

    #         total = (1.0 - self.beta) * inv_loss + self.beta * fwd_loss
    #         return total.astype(jnp.float32), {
    #             "icm_inv_loss": inv_loss,
    #             "icm_fwd_loss": fwd_loss,
    #         }

    #     opt_mets, aux = self.opt([self.inv, self.fwd], loss_fn, has_aux=True)
    #     return {**opt_mets, **aux}


# class ICM(nj.Module):
#     def __init__(self, wm, act_space, config, name="icm"):
#         self.act_dim = int(np.prod(act_space.shape))
#         self.discrete = act_space.discrete
#         if not self.discrete:
#             raise NotImplementedError("ICM inverse model assumes discrete one-hot actions.")
#         self.beta = config.icm_beta          # loss blend
#         self.eta  = getattr(config, "icm_eta", 1.0)  # reward scale, separate from beta
#         # self.act_dim   = int(np.prod(act_space.shape))
#         # self.beta      = config.icm_beta

#         # Inverse model should predict a categorical over discrete actions.
#         inv_kw = dict(config.icm_head)
#         inv_kw["dist"] = "onehot"
#         self.inv = nets.MLP(shape=(self.act_dim,), **inv_kw, name="inv")

#         # Forward model predicts next latent vector; we set shape at runtime
#         # to match z_tp1 last-dim so it returns a distribution with .mean().
#         self.fwd = nets.MLP(shape=None, **config.icm_head, name="fwd")
#         self.opt = jaxutils.Optimizer(name="icm_opt", **config.icm_opt)
#         self.mom = jaxutils.Moments(**config.icm_norm, name="icm_norm")

#     def __call__(self, traj):
#         def _flat(stoch):
#             return stoch.reshape(stoch.shape[:2] + (-1,))
#         z  = jnp.concatenate([traj["deter"], _flat(traj["stoch"])], -1)  
#         z_t, z_tp1 = z[:-1], z[1:]              
#         act_t      = traj["action"][:-1] 

#         self.fwd._shape = z_tp1.shape[2:]
#         pred = self.fwd(jnp.concatenate([z_t, act_t], -1)).mean()
#         fwd_err = 0.5 * jnp.square(pred - z_tp1).mean(-1)   # [T,B]
#         mean, invstd = self.mom(fwd_err)
#         return self.eta * (fwd_err - mean) * invstd         # do NOT use beta here     

#     def train(self, data):
#         flat = lambda x: x.reshape(x.shape[:2] + (-1,))
#         z  = jnp.concatenate([data["deter"], flat(data["stoch"])], -1)
#         a  = data["action"]
#         # stop-grad to avoid updating WM via ICM training
#         z, a = jax.lax.stop_gradient(z), jax.lax.stop_gradient(a)

#         # time-major
#         z, a = jnp.swapaxes(z, 0, 1), jnp.swapaxes(a, 0, 1)   # [T+1,B,...], [T,B,A]
#         T = z.shape[0] - 1
#         z_t, z_tp1, a_t = z[:T], z[1:], a[:T]

#         # mask across episode boundaries if available
#         if "cont" in data:
#             cont = jnp.swapaxes(data["cont"][:, 1:], 0, 1)    # [T,B]
#         else:
#             cont = jnp.ones(a_t.shape[:2], jnp.float32)

#         inv_dist = self.inv(jnp.concatenate([z_t, z_tp1], -1))
#         inv_logp = inv_dist.log_prob(a_t)                     # [T,B]
#         inv_loss_step = -inv_logp

#         self.fwd._shape = z_tp1.shape[2:]
#         pred = self.fwd(jnp.concatenate([z_t, a_t], -1)).mean()
#         fwd_loss_step = jnp.square(pred - z_tp1).mean(-1)     # [T,B]

#         denom = jnp.maximum(cont.sum(), 1.0)
#         inv_loss = (inv_loss_step * cont).sum() / denom
#         fwd_loss = (fwd_loss_step * cont).sum() / denom

#         total = (1.0 - self.beta) * inv_loss + self.beta * fwd_loss

        

#         return self.opt([self.inv, self.fwd], lambda: (total, {
#             "icm_inv_loss": inv_loss,
#             "icm_fwd_loss": fwd_loss,
#         }), has_aux=True)        






    # def train(self, data):
    #     flat = lambda x: x.reshape(x.shape[:2] + (-1,))
    #     z  = jnp.concatenate([data["deter"], flat(data["stoch"])], -1) 
    #     a  = data["action"]   
    #     # stop-grad to avoid updating WM via ICM training
    #     z, a = jax.lax.stop_gradient(z), jax.lax.stop_gradient(a)  
    #     T = z.shape[0] - 1
    #     z_t, z_tp1, a_t = z[:T], z[1:], a[:T]
                                      
    #     # z, a = jnp.swapaxes(z, 0, 1), jnp.swapaxes(a, 0, 1)  
    #     #           
    #     T = min(z.shape[0] - 1, a.shape[0])            
    #     if T == 0:                                      
    #         return {"icm_inv_loss": 0.0, "icm_fwd_loss": 0.0}

    #     z, a = z[:T + 1], a[:T]                         
    #     z_t, z_tp1 = z[:-1], z[1:]

        
    #     def loss_fn():
    #         # Inverse loss: onehot distribution over actions to match one-hot 'a'.
    #         inv_dist   = self.inv(jnp.concatenate([z_t, z_tp1], -1))
    #         inv_loss   = -inv_dist.log_prob(a).mean()

    #         # Forward loss: set output shape to z_tp1 to predict next latent.
    #         self.fwd._shape = z_tp1.shape[2:]
    #         pred_z_tp1 = self.fwd(jnp.concatenate([z_t, a], -1)).mean()
    #         fwd_loss   = jnp.square(pred_z_tp1 - z_tp1).mean()

    #         return inv_loss + self.beta * fwd_loss, {
    #             "icm_inv_loss": inv_loss,
    #             "icm_fwd_loss": fwd_loss,
    #         }

    #     opt_mets, aux_mets = self.opt([self.inv, self.fwd], loss_fn, has_aux=True)

    #     return {**opt_mets, **aux_mets}
