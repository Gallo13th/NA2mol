"""Patch v2: two-channel inpainting conditioning with static (non-fluctuating) beliefs.

Design (per reviewer):
  * three conditioning tasks share one parameterisation: per atom, a *position* mask and a
    *type* mask say which attribute is given;
      - fix ring system  -> pos+type fixed on ring atoms, inpaint linkers/substituents (primary)
      - fix types        -> type fixed, positions generated (docking-like)
      - fix positions    -> position fixed, types generated (interaction refine)
  * the fixed attributes enter the network as *constants*: the position belief of a
    position-fixed atom is its true coordinate at every step, the type belief of a
    type-fixed atom is its one-hot, i.e. no forward-noise, no trajectory fluctuation;
  * the loss is computed only on the free attribute (masked per atom), so the model is
    never asked to reproduce what it was given.

Run from the project root after restoring model/bfn4edmol.py from .bak_preinpaint:
    cp model/bfn4edmol.py.bak_preinpaint model/bfn4edmol.py
    python train_scripts/patch_model_for_inpaint.py
"""
import pathlib

P = pathlib.Path('model/bfn4edmol.py')
src = P.read_text()

# ---------------- 1. two extra input channels -------------------------------------
a = """        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )"""
b = """        # +2: per-atom inpainting indicators (position given?, type given?). The extra
        # input columns are zeroed after checkpoint loading, so the pretrained model is intact.
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim + 2, emb_dim
        )"""
assert src.count(a) == 1
src = src.replace(a, b)

# ---------------- 2. interdependency_modeling signature ----------------------------
a = """        batch_ligand,  # index for ligand
        gamma_coord,
        return_all=False,  # legacy from targetdiff"""
b = """        batch_ligand,  # index for ligand
        gamma_coord,
        pos_mask=None,   # [N_ligand] bool: this atom's position is given (constant belief)
        type_mask=None,  # [N_ligand] bool: this atom's element type is given (constant belief)
        return_all=False,  # legacy from targetdiff"""
assert src.count(a) == 1
src = src.replace(a, b)

# ---------------- 3. feed the two indicators as node features ----------------------
a = """        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v
"""
b = a + """        n_lig = input_ligand_feat.size(0)
        if pos_mask is None:
            pos_mask = torch.zeros(n_lig, dtype=torch.bool, device=input_ligand_feat.device)
        if type_mask is None:
            type_mask = torch.zeros(n_lig, dtype=torch.bool, device=input_ligand_feat.device)
        input_ligand_feat = torch.cat(
            [input_ligand_feat, pos_mask.float().view(-1, 1), type_mask.float().view(-1, 1)], -1)
"""
assert src.count(a) == 1
src = src.replace(a, b)

# ---------------- 4. loss_one_step: static beliefs + per-attribute masking ----------
a = """        batch_ligand=None,
        # batch_ligand_ei=None
    ):
        K = self.num_classes"""
b = """        batch_ligand=None,
        pos_mask=None,
        type_mask=None,
        # batch_ligand_ei=None
    ):
        K = self.num_classes"""
assert src.count(a) == 1
src = src.replace(a, b)

a = """        # discrete ~ N(y | β(t)(Ke_x−1), β(t)KI)
        theta = self.discrete_var_bayesian_update(
            t, beta1=self.beta1, x=ligand_v, K=K
        )  # [N, K]
"""
b = a + """        # fixed attributes enter as constants: no forward noise on what we were given
        if pos_mask is not None:
            mu_coord = torch.where(pos_mask.unsqueeze(-1), ligand_pos, mu_coord)
        if type_mask is not None:
            theta = torch.where(type_mask.unsqueeze(-1), ligand_v, theta)
"""
assert src.count(a) == 1
src = src.replace(a, b)

# pass the masks into the network call inside loss_one_step
a = """            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
        )  # [N, 3], [N, K], [?]"""
b = """            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
            pos_mask=pos_mask,
            type_mask=type_mask,
        )  # [N, 3], [N, K], [?]"""
assert src.count(a) == 1
src = src.replace(a, b)

# ---------------- 5. mask the loss to the free attribute ---------------------------
a = """        # 3. Compute reweighted loss (previous [N,] now [B,])
        if not self.use_discrete_t:"""
b = """        # 3. Compute reweighted loss (previous [N,] now [B,]), restricted to free attributes:
        # atoms whose attribute was given are pushed into a sentinel segment and dropped after.
        n_graphs = int(batch_ligand.max().item()) + 1
        seg_coord = batch_ligand.clone()
        if pos_mask is not None:
            seg_coord = torch.where(pos_mask, torch.full_like(batch_ligand, n_graphs), seg_coord)
        seg_type = batch_ligand.clone()
        if type_mask is not None:
            seg_type = torch.where(type_mask, torch.full_like(batch_ligand, n_graphs), seg_type)

        if not self.use_discrete_t:"""
assert src.count(a) == 1
src = src.replace(a, b)

# swap the segment ids in the two loss branches and slice off the sentinel row
a = """                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]
            # print(dloss.shape)"""
assert src.count(a) == 1
src = src.replace(a, """                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=seg_coord,
            )[:n_graphs]  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=seg_type,
            )[:n_graphs]  # [B,]""")

a = """            closs = self.dtime4continuous_loss(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )"""
assert src.count(a) == 1
src = src.replace(a, """            closs = self.dtime4continuous_loss(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=seg_coord,
            )[:n_graphs]""")

a = """            dloss = self.dtime4discrete_loss_prob(
                i=i,
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )
"""
assert src.count(a) == 1
src = src.replace(a, """            dloss = self.dtime4discrete_loss_prob(
                i=i,
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v,
                p_0=p0_h,
                K=K,
                segment_ids=seg_type,
            )[:n_graphs]
""", 1)

# ---------------- 6. sample_by_seed: constant beliefs + two masks -------------------
a = """                mu_pos_input = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_mu_pos[i - 1], mu_pos_t)
                if fix_atom_types:
                    theta_h_input = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_theta_h[i - 1], theta_h_t)
                else:
                    theta_h_input = theta_h_t"""
assert src.count(a) == 1
src = src.replace(a, """                # fixed attributes are constants, not a noisy trajectory
                mu_pos_input = torch.where(pos_fixed_mask.unsqueeze(-1), fixed_atom_pos_full, mu_pos_t)
                if fix_atom_types:
                    theta_h_input = torch.where(type_fixed_mask.unsqueeze(-1),
                                                fixed_atom_types_full, theta_h_t)
                else:
                    theta_h_input = theta_h_t""")

a = """                batch_ligand=batch_ligand,
                theta_h_t=theta_h_input,
                mu_pos_t=mu_pos_input,"""
assert src.count(a) == 1
src = src.replace(a, """                batch_ligand=batch_ligand,
                pos_mask=pos_fixed_mask,
                type_mask=type_fixed_mask,
                theta_h_t=theta_h_input,
                mu_pos_t=mu_pos_input,""")

# ---------------- 7. optional type-only mask in sample_by_seed ----------------------
a = """        sample_steps=1000,"""
assert src.count(a) >= 1
# insert the optional type mask among the keyword arguments (it must not precede required ones)
first = src.index(a)
src = src[:first] + """        sample_steps=1000,
        type_fixed_mask=None,  # [N_ligand_total], bool (element types fixed)""" + src[first + len(a):]

a = """        free_mask = ~fixed_atom_mask  # [N_ligand_total]"""
assert src.count(a) == 1
src = src.replace(a, """        if type_fixed_mask is None:
            type_fixed_mask = fixed_atom_mask
        pos_fixed_mask = fixed_atom_mask
        free_mask = ~fixed_atom_mask  # [N_ligand_total]""")

P.write_text(src)
print('patched v2:', P)
