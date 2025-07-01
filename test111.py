import jax.numpy as jnp
import numpy as np
import functools
def slogdet(x):
  """Computes sign and log of determinants of matrices.

  This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

  Args:
    x: square matrix.

  Returns:
    sign, (natural) logarithm of the determinant of x.
  """
  if x.shape[-1] == 1:
    sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet
# Sample data: list of square matrices
xs = [
    np.array([[4, 1], [1, 4]]),  # 2x2 matrix, det = 15
    np.array([[2, 0], [0, 3]]),  # 2x2 matrix, det = 6
    np.array([[5]]),              # 1x1 matrix, will be skipped
    np.array([[1, 2], [3, 4]])   # 2x2 matrix, det = -2
]

# The core computation
sign_in, logdet = functools.reduce(
    lambda a, b: (a[0] * b[0], a[1] + b[1]),
    [slogdet(x) for x in xs if x.shape[-1] > 1],
    (1, 0)
)
print("sign_in:",sign_in)
print("logdet:",logdet)
# Compute the determinant from sign and logdet
determinant = sign_in * np.exp(logdet)

# Print results for clarity
print("Input matrices (excluding 1x1):")
for x in xs:
    if x.shape[-1] > 1:
        print(x)
        print(f"Determinant: {np.linalg.det(x):.4f}\n")

print(f"Product of signs: {sign_in}")
print(f"Sum of log-determinants: {logdet:.4f}")
print(f"Total determinant (sign * exp(logdet)): {determinant:.4f}")

# Verify by computing product of determinants directly
dets = [np.linalg.det(x) for x in xs if x.shape[-1] > 1]
prod_dets = np.prod(dets)
print(f"Product of determinants (direct): {prod_dets:.4f}")