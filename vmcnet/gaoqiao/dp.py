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


class ManyElectronSystem():
  def __init__(
      self,
      charges,
      nspins,
  ):
    self.natoms = charges.shape[0]
    self.nelecs = sum(nspins)
    self.nparts = self.natoms + self.nelecs
    self.np_spin = list(nspins) + [self.natoms]
    self.np = [self.nelecs, self.natoms]
    self.charges = np.array(charges, dtype = np.int32)

    self.uniq_charges = np.unique(np.sort(self.charges))
    self.n_uniq_charges = self.uniq_charges.size
    self.types = np.zeros(self.natoms, dtype = np.int32)
    for ii in range(len(self.uniq_charges)):
      self.types += (charges == self.uniq_charges[ii]) * ii
    self.non_zero_spin_channels = np.sum(np.array(nspins,dtype=int) != 0)
    self.types += self.non_zero_spin_channels
    self.types = jnp.concatenate([
        np.zeros(nspins[0]), np.ones(nspins[1]), self.types])
    
    self.dim_one_hot = self.n_uniq_charges + self.non_zero_spin_channels
    self.type_one_hot = jax.nn.one_hot(self.types, self.dim_one_hot)
    ta = self.type_one_hot
    self.pair_one_hot = jnp.concatenate([
        jnp.tile(ta.reshape([self.nparts, 1, -1]), [1, self.nparts, 1]),
        jnp.tile(ta.reshape([1, self.nparts, -1]), [self.nparts, 1, 1]),
    ], axis=-1)

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
    ea_split = self.get_split_ea()[:-1]
    split0 = jnp.split(data, ea_split, axis=axis[0])
    [ee, ea] = jnp.split(split0[0], ea_split, axis=axis[1])
    [ae, aa] = jnp.split(split0[1], ea_split, axis=axis[1])
    return ee, ea, ae, aa
