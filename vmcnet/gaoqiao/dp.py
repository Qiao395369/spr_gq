import jax
import numpy as np
import jax.numpy as jnp

def apply_pbc(
        rr,
        lattice,
        rec_lattice,
):
    shape = rr.shape
    sr = jnp.matmul(rr.reshape(-1,3), rec_lattice)
    sr = jnp.mod(sr, 1.0)
    return jnp.matmul(sr, lattice).reshape(shape)

def apply_nearest_neighbor(
        rij,
        lattice,
        rec_lattice,
):
    srij = jnp.matmul(rij, rec_lattice)
    srij = jnp.mod(srij+0.5, 1.0) - 0.5
    rij = jnp.matmul(srij, lattice)
    return rij

def auto_nearest_neighbor(
        lattice,
        rc,
) -> bool:
    vol = np.linalg.det(lattice)
    tofacedist = np.cross(lattice[[1,2,0],:], lattice[[2,0,1],:])
    tofacedist = vol * np.reciprocal(np.linalg.norm(tofacedist, axis=1))
    return (rc <= 0.5 * tofacedist[0]) and (rc <= 0.5 * tofacedist[1]) and (rc <= 0.5 * tofacedist[2])


def spline_func(xx, rc, rc_smth):
    uu = (xx - rc_smth) / (rc - rc_smth)
    return uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1

def switch_func_poly(
        xx,
        rc = 3.0,
        rc_smth = 0.2,
):
    ret = \
        1.0 * (xx < rc_smth) + \
        spline_func(xx, rc, rc_smth) * jnp.logical_and(xx >= rc_smth, xx < rc) + \
        0.0 * (xx >= rc)
    return ret


class ManyElectronSystem_old():
  def __init__(
      self,     #以CONH3为例
      charges,  #[6, 8, 7, 1, 1, 1]
      nspins,   #(12,12)
  ):
    self.natoms = charges.shape[0]   #6
    self.nelecs = sum(nspins)        #24
    self.nparts = self.natoms + self.nelecs             #30
    self.np_spin = list(nspins) + [self.natoms]         #[12,12,6]
    self.np = [self.nelecs, self.natoms]                #[24,6]
    self.charges = np.array(charges, dtype = np.int64)  #[6, 8, 7, 1, 1, 1]

    self.uniq_charges = np.unique(np.sort(self.charges))  #[1,6,7,8]   (得到升序、无重复的电荷列表)
    self.n_uniq_charges = self.uniq_charges.size          #4
    self.types = np.zeros(self.natoms, dtype = np.int64)  #[0,0,0,0,0,0]
    for ii in range(len(self.uniq_charges)):
      self.types += (charges == self.uniq_charges[ii]) * ii  #[1,3,2,0,0,0]
    self.non_zero_spin_channels = np.sum(np.array(nspins,dtype=int) != 0)  #2
    self.types += self.non_zero_spin_channels                 #[3,5,4,2,2,2]
    self.types = jnp.concatenate([
        np.zeros(nspins[0]), np.ones(nspins[1]), self.types])  #[0,...0,1,...,1,3,5,4,2,2,2]
    
    self.dim_one_hot = self.n_uniq_charges + self.non_zero_spin_channels  #4+2=6 
    self.type_one_hot = jax.nn.one_hot(self.types, self.dim_one_hot)#dim_one_hot:类别总数(独热向量长度) type_one_hot:(30,6)
    ta = self.type_one_hot
    self.pair_one_hot = jnp.concatenate([
        jnp.tile(ta.reshape([self.nparts, 1, -1]), [1, self.nparts, 1]),  #(30,6)->(30,1,6)->(30,30,6)
        jnp.tile(ta.reshape([1, self.nparts, -1]), [self.nparts, 1, 1]),  #(30,6)->(1,30,6)->(30,30,6)
    ], axis=-1)     #->(30,30,12)

  def get_dim_one_hot(self):
    return self.dim_one_hot

  def get_part_one_hot(self):
    return self.type_one_hot

  def get_pair_one_hot(self):
    return self.pair_one_hot

  def get_split_ea(self):
    return self.np  #[self.nelecs, self.natoms]

  def get_split_eea(self):
    return self.np_spin

  def split_ea(self, data, axis=0):
    return jnp.split(data, self.get_split_ea()[:-1], axis=axis)

  def split_ee_ea_aa(self, data, axis=(0,1)):
    """
    data : np x np x ...
    """
    ea_split = self.get_split_ea()[:-1]  #[nele]
    split0 = jnp.split(data, ea_split, axis=axis[0])  #(np,np,...)->(nele,np,...),(na,np,...)
    [ee, ea] = jnp.split(split0[0], ea_split, axis=axis[1])  #(nele,np,...)->(nele,nele,...),(nele,na,...)
    [ae, aa] = jnp.split(split0[1], ea_split, axis=axis[1])  #(na,np,...)->(na,nele,...),(na,na,...)
    return ee, ea, aa

  def split_ee_ea_ae_aa(self, data, axis=(0,1)):
    """
    data : np x np x ...
    """
    ea_split = self.get_split_ea()[:-1]  #[nele]
    split0 = jnp.split(data, ea_split, axis=axis[0])  #(np,np,...)->(nele,np,...),(na,np,...)
    [ee, ea] = jnp.split(split0[0], ea_split, axis=axis[1])  #(nele,np,...)->(nele,nele,...),(nele,na,...)
    [ae, aa] = jnp.split(split0[1], ea_split, axis=axis[1])  #(na,np,...)->(na,nele,...),(na,na,...)
    return ee, ea, ae, aa
  
def split_ee_ea_ae_aa_(ne, data, axis=(0,1)):
    """
    data : np x np x ...
    """
    ea_split = [ne] #[nele]
    split0 = jnp.split(data, ea_split, axis=axis[0])  #(np,np,...)->(nele,np,...),(na,np,...)
    [ee, ea] = jnp.split(split0[0], ea_split, axis=axis[1])  #(nele,np,...)->(nele,nele,...),(nele,na,...)
    [ae, aa] = jnp.split(split0[1], ea_split, axis=axis[1])  #(na,np,...)->(na,nele,...),(na,na,...)
    return ee, ea, ae, aa

def reform_ee_ea_ae_aa(ee, ea, ae, aa):
    ee_ea=jnp.concatenate([ee,ea],axis=1)
    ae_aa=jnp.concatenate([ae,aa],axis=1)
    return jnp.concatenate([ee_ea,ae_aa],axis=0)


INT=np.int64
class ManyElectronSystem:
  """
  dp_type:
    "original":
        particle feature:
          [onehot_up, onehot_down, onehot_nucleus]

        pair feature:
          concat(feature_i, feature_j)

    "element":
        particle feature:
          [onehot_up, onehot_down, onehot_element_1, onehot_element_2, ...]

        Element order follows first appearance in charges.
        Example: C2H6 with charges [6, 6, 1, 1, 1, 1, 1, 1]
          -> [onehot_up, onehot_down, onehot_C, onehot_H]

        pair feature:
          concat(feature_i, feature_j)

    "charge":
        particle feature:
          [onehot_up, onehot_down, onehot_nucleus, q_norm]

        q_norm:
          electron = -1 / Zmax
          nucleus  =  Z / Zmax

        pair feature:
          [onehot_i, onehot_j, qi, qj, qi*qj]

    "full":
        particle feature:
          [onehot_up, onehot_down, onehot_nucleus,
           q, |q|, q^2, sign(q), spin_marker]

        spin_marker:
          up electron   = +1
          down electron = -1
          nucleus       =  0

        pair feature:
          [onehot_i, onehot_j,
           qi, qj, qi*qj, |qi*qj|,
           si, sj, si*sj]

  Notes:
    - get_part_one_hot() and get_pair_one_hot() are kept for compatibility,
      but they return the active descriptor controlled by dp_type.
    - If you need the raw 3-way one-hot, use get_raw_part_one_hot().
  """

  def __init__(
      self,
      charges,
      nspins,
      dp_type: str = "original",
  ):
    charges = np.asarray(charges, dtype=INT)
    nspins = tuple(int(x) for x in nspins)

    if len(nspins) != 2:
      raise ValueError(f"Expected nspins=(n_up, n_down), got {nspins}")

    if dp_type not in ("original", "charge", "full", "element","element1"):
      raise ValueError(f"dp_type must be one of {'original', 'charge', 'full', 'element','element1'}, got {dp_type}")

    self.dp_type = dp_type
    self.natoms = int(charges.shape[0])
    self.nelecs = int(sum(nspins))
    self.nparts = self.natoms + self.nelecs
    self.nspins = nspins
    self.np_spin = list(nspins) + [self.natoms]
    self.np = [self.nelecs, self.natoms]
    self.charges = np.array(charges, dtype=INT)

    # ------------------------------------------------------------
    # 1. Raw 3-way particle type:
    #      0: spin-up electron
    #      1: spin-down electron
    #      2: nucleus
    # ------------------------------------------------------------
    raw_types = np.concatenate([np.zeros(nspins[0],dtype=INT),np.ones(nspins[1],dtype=INT),2*np.ones(self.natoms,dtype=INT)],axis=0)
    self.raw_types = jnp.asarray(raw_types, dtype=jnp.int64)

    self.raw_dim_one_hot = 3
    self.raw_part_one_hot = jax.nn.one_hot(self.raw_types, self.raw_dim_one_hot)
    self.raw_pair_one_hot = self._pair_concat(self.raw_part_one_hot)

    # ------------------------------------------------------------
    # 2. Physical charge q
    #
    #    electrons: -1
    #    nuclei:     Z
    #
    #    normalized by max nuclear charge Zmax
    # ------------------------------------------------------------
    electron_charges = -np.ones(self.nelecs, dtype=np.float64)
    nuclear_charges = self.charges.astype(np.float64)
    part_charges = np.concatenate([electron_charges, nuclear_charges], axis=0)  #(np,)

    zmax = float(np.max(np.abs(nuclear_charges))) if self.natoms > 0 else 1.0
    if zmax <= 0.0:
      zmax = 1.0

    self.charge_scale = zmax
    self.part_charges = jnp.asarray(part_charges, dtype=jnp.float64)
    self.part_charges_norm = self.part_charges / self.charge_scale
    q = self.part_charges_norm      #(np,)

    # ------------------------------------------------------------
    # 3. Spin marker:
    #      up electron   = +1
    #      down electron = -1
    #      nucleus       =  0
    # ------------------------------------------------------------
    spin_marker = np.concatenate([np.ones(nspins[0],dtype=np.float64), -np.ones(nspins[1],dtype=np.float64), np.zeros(self.natoms,dtype=np.float64)], axis=0)
    self.spin_marker = jnp.asarray(spin_marker, dtype=jnp.float64)

    # ------------------------------------------------------------
    # 4. Active particle / pair descriptors
    # ------------------------------------------------------------
    if dp_type == "original":
      self.part_one_hot = self.raw_part_one_hot
      self.pair_one_hot = self.raw_pair_one_hot

    elif dp_type == "element":
      # particle: element-wise one-hot.
      # Example C2H6 with charges [6, 6, 1, 1, 1, 1, 1, 1]:
      #   [up, down, C, H]
      # Element order follows first appearance in self.charges.
      element_charges = []
      for z in self.charges:
        z_int = int(z)
        if z_int not in element_charges:
          element_charges.append(z_int)
      self.element_charges = np.asarray(element_charges, dtype=INT)

      atom_types = np.zeros(self.natoms, dtype=INT)
      for ii, z in enumerate(self.element_charges):
        atom_types += (self.charges == z).astype(INT) * (2 + ii)

      element_types = np.concatenate([
          np.zeros(nspins[0], dtype=INT),
          np.ones(nspins[1], dtype=INT),
          atom_types,
      ], axis=0)
      self.element_types = jnp.asarray(element_types, dtype=jnp.int64)
      self.element_dim_one_hot = 2 + len(self.element_charges)

      self.part_one_hot = jax.nn.one_hot(self.element_types, self.element_dim_one_hot)
      self.pair_one_hot = self._pair_concat(self.part_one_hot)

    elif dp_type == "element1":
      # Same output convention as the first ManyElectronSystem:
      # element order = sorted unique nuclear charges.
      # Example charges [6, 7, 8, 1, 1, 1]
      # -> element order [1, 6, 7, 8]
      # -> [onehot_up, onehot_down, onehot_H, onehot_C, onehot_N, onehot_O]

      self.uniq_charges = np.unique(np.sort(self.charges)).astype(INT)
      self.n_uniq_charges = self.uniq_charges.size

      atom_types = np.zeros(self.natoms, dtype=INT)
      for ii in range(len(self.uniq_charges)):
        atom_types += (self.charges == self.uniq_charges[ii]).astype(INT) * ii

      self.non_zero_spin_channels = np.sum(np.array(nspins, dtype=int) != 0)
      atom_types += self.non_zero_spin_channels

      element1_types = jnp.concatenate([
          np.zeros(nspins[0], dtype=INT),
          np.ones(nspins[1], dtype=INT),
          atom_types,
      ], axis=0)

      self.element_charges = self.uniq_charges
      self.element1_charges = self.uniq_charges
      self.element1_types = jnp.asarray(element1_types, dtype=jnp.int64)
      self.element1_dim_one_hot = self.n_uniq_charges + self.non_zero_spin_channels

      self.part_one_hot = jax.nn.one_hot(
          self.element1_types,
          self.element1_dim_one_hot,
      )
      self.pair_one_hot = self._pair_concat(self.part_one_hot)

      # Compatibility aliases, matching the naming style of the first code.
      self.types = self.element1_types
      self.type_one_hot = self.part_one_hot

    elif dp_type == "charge":
      # particle: [3-way one-hot, q]
      self.part_one_hot = jnp.concatenate([self.raw_part_one_hot , q[:, None],], axis=-1)

      qi = jnp.broadcast_to(q[:, None], (self.nparts, self.nparts))
      qj = jnp.broadcast_to(q[None, :], (self.nparts, self.nparts))
      qiqj = qi * qj

      # pair: [raw onehot_i, raw onehot_j, qi, qj, qi*qj]
      self.pair_one_hot = jnp.concatenate([self.raw_pair_one_hot, qi[..., None], qj[..., None], qiqj[..., None],], axis=-1)

    elif dp_type == "full":
      s = self.spin_marker

      q_features = jnp.stack([q, jnp.abs(q), q ** 2, jnp.sign(q)], axis=-1)

      # particle:
      # [onehot_up/down/nucleus, q, |q|, q^2, sign(q), spin_marker]
      self.part_one_hot = jnp.concatenate([self.raw_part_one_hot, q_features, s[:, None]], axis=-1)

      qi = jnp.broadcast_to(q[:, None], (self.nparts, self.nparts))
      qj = jnp.broadcast_to(q[None, :], (self.nparts, self.nparts))
      qiqj = qi * qj

      si = jnp.broadcast_to(s[:, None], (self.nparts, self.nparts))
      sj = jnp.broadcast_to(s[None, :], (self.nparts, self.nparts))
      sisj = si * sj

      # pair:
      # [raw onehot_i, raw onehot_j, qi, qj, qi*qj, |qi*qj|, si, sj, si*sj]
      self.pair_one_hot = jnp.concatenate([
          self.raw_pair_one_hot, qi[..., None], qj[..., None], qiqj[..., None], 
          jnp.abs(qiqj)[..., None], si[..., None], sj[..., None], sisj[..., None],
      ], axis=-1)

    self.dim_one_hot = int(self.part_one_hot.shape[-1])
    self.dim_pair_one_hot = int(self.pair_one_hot.shape[-1])

  # ------------------------------------------------------------
  # Internal helper
  # ------------------------------------------------------------
  def _pair_concat(self, part_feat):
    """
    part_feat: [nparts, dim]
    return:    [nparts, nparts, 2 * dim]
    """
    return jnp.concatenate([
        jnp.tile(part_feat.reshape([self.nparts, 1, -1]), [1, self.nparts, 1]),
        jnp.tile(part_feat.reshape([1, self.nparts, -1]), [self.nparts, 1, 1]),
    ], axis=-1)

  # ------------------------------------------------------------
  # Active feature API
  # ------------------------------------------------------------
  def get_dim_one_hot(self):
    """ Active particle descriptor dimension. """
    return self.dim_one_hot

  def get_dim_pair_one_hot(self):
    """ Active pair descriptor dimension. """
    return self.dim_pair_one_hot

  def get_part_one_hot(self):
    """ Kept for compatibility.Returns active particle descriptor controlled by dp_type. """
    return self.part_one_hot

  def get_pair_one_hot(self):
    """ Kept for compatibility.Returns active pair descriptor controlled by dp_type. """
    return self.pair_one_hot

  # aliases, if you prefer clearer names
  def get_part_features(self):
    return self.part_one_hot

  def get_pair_features(self):
    return self.pair_one_hot

  def get_dim_part_features(self):
    return self.dim_one_hot

  def get_dim_pair_features(self):
    return self.dim_pair_one_hot

  # ------------------------------------------------------------
  # Raw original 3-way one-hot API
  # ------------------------------------------------------------
  def get_raw_dim_one_hot(self):
    return self.raw_dim_one_hot

  def get_raw_part_one_hot(self):
    return self.raw_part_one_hot

  def get_raw_pair_one_hot(self):
    return self.raw_pair_one_hot

  # ------------------------------------------------------------
  # Charge / spin API
  # ------------------------------------------------------------
  def get_part_charges(self):
    """ Unnormalized physical charges: electron = -1  nucleus = Z """
    return self.part_charges

  def get_part_charges_norm(self):
    """ Normalized charges: electron = -1 / Zmax  nucleus =  Z / Zmax """
    return self.part_charges_norm

  def get_spin_marker(self):
    """ up electron = +1   down electron = -1   nucleus = 0 """
    return self.spin_marker

  # ------------------------------------------------------------
  # Split API
  # ------------------------------------------------------------
  def get_split_ea(self):
    return self.np

  def get_split_eea(self):
    return self.np_spin

  def split_ea(self, data, axis=0):
    """
    Split particle data into electron part and atom part.

    data shape example:
      [nelec + natom, ...]
    """
    return jnp.split(data, self.get_split_ea()[:-1], axis=axis)

  def split_ee_ea_aa(self, data, axis=(0, 1)):
    """
    data: [nparts, nparts, ...]

    return:
      ee: [nelec, nelec, ...]
      ea: [nelec, natom, ...]
      aa: [natom, natom, ...]
    """
    ea_split = self.get_split_ea()[:-1]

    split0 = jnp.split(data, ea_split, axis=axis[0])

    ee, ea = jnp.split(split0[0], ea_split, axis=axis[1])
    ae, aa = jnp.split(split0[1], ea_split, axis=axis[1])

    return ee, ea, aa

  def split_ee_ea_ae_aa(self, data, axis=(0, 1)):
    """
    data: [nparts, nparts, ...]

    return:
      ee: [nelec, nelec, ...]
      ea: [nelec, natom, ...]
      ae: [natom, nelec, ...]
      aa: [natom, natom, ...]
    """
    ea_split = self.get_split_ea()[:-1]

    split0 = jnp.split(data, ea_split, axis=axis[0])

    ee, ea = jnp.split(split0[0], ea_split, axis=axis[1])
    ae, aa = jnp.split(split0[1], ea_split, axis=axis[1])

    return ee, ea, ae, aa

  def print_summary(self):
    print("ManyElectronSystem summary")
    print("  dp_type:", self.dp_type)
    print("  natoms:", self.natoms)
    print("  nelecs:", self.nelecs)
    print("  nparts:", self.nparts)
    print("  charges:", self.charges)
    print("  nspins:", self.nspins)
    print("  charge_scale:", self.charge_scale)
    print("  raw_dim_one_hot:", self.raw_dim_one_hot)
    print("  dim_one_hot:", self.dim_one_hot)
    print("  dim_pair_one_hot:", self.dim_pair_one_hot)
    print("  part_one_hot shape:", self.part_one_hot.shape)
    print("  pair_one_hot shape:", self.pair_one_hot.shape)
    print("  part_one_hot:", self.part_one_hot)
    print("  pair_one_hot[0]:", self.pair_one_hot[0])

if __name__ == "__main__":
    # 测试用参数：C2H6 分子（2个C，6个H），自旋向上2个电子，向下2个电子
    test_charges = [6, 7, 8,1,1,1]  # 核电荷
    test_nspins = (12, 12)  # 上自旋电子数，下自旋电子数
    
    # 测试所有4种类型
    test_types = ["original", "element", "charge", "full"]
    
    for dp_type in test_types:
        print("=" * 80)
        print(f"测试 dp_type = {dp_type}")
        print("=" * 80)
        
        # 1. 初始化系统
        system = ManyElectronSystem(
            charges=test_charges,
            nspins=test_nspins,
            dp_type=dp_type
        )
        
        # 2. 打印系统总结
        system.print_summary()
        print("\n" + "-" * 50)
        
        # 3. 测试核心API方法
        print("=== 测试核心API方法 ===")
        # 粒子/对特征
        part_feat = system.get_part_features()
        pair_feat = system.get_pair_features()
        print(f"粒子特征形状: {part_feat.shape}")
        print(f"成对特征形状: {pair_feat.shape}")
        
        # 维度
        dim_part = system.get_dim_part_features()
        dim_pair = system.get_dim_pair_features()
        print(f"粒子特征维度: {dim_part}")
        print(f"成对特征维度: {dim_pair}")
        
        # 电荷与自旋
        charges = system.get_part_charges()
        charges_norm = system.get_part_charges_norm()
        spin_marker = system.get_spin_marker()
        print(f"物理电荷前5个: {charges[:5]}")
        print(f"归一化电荷前5个: {charges_norm[:5]}")
        print(f"自旋标记前5个: {spin_marker[:5]}")
        
        # 原始独热编码
        raw_part = system.get_raw_part_one_hot()
        print(f"原始3分类独热形状: {raw_part.shape}")
        
        # 4. 测试分割功能
        print("\n=== 测试数据分割功能 ===")
        # 粒子级分割（电子/原子核）
        elec_part, atom_part = system.split_ea(part_feat)
        print(f"电子特征形状: {elec_part.shape}")
        print(f"原子核特征形状: {atom_part.shape}")
        
        # 成对特征四分割 (ee, ea, ae, aa)
        ee, ea, ae, aa = system.split_ee_ea_ae_aa(pair_feat)
        print(f"电子-电子成对特征: {ee.shape}")
        print(f"电子-原子核成对特征: {ea.shape}")
        print(f"原子核-电子成对特征: {ae.shape}")
        print(f"原子核-原子核成对特征: {aa.shape}")
        
        print("\n" + "=" * 80 + "\n\n")
    
    print("✅ 所有类型测试完成！")